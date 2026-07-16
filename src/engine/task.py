from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from src.metrics.iou import SIOU, VIOU

# Stateless + cheap (sign compare on the SDF already in hand, no MC/KDTree) → log every step.
_VIOU, _SIOU = VIOU(), SIOU()


@dataclass
class StepOutput:
    """Result of a single forward step.

    Args:
        loss: scalar tensor to backprop.
        metrics: extra scalars to log (e.g. {"kl": ..., "recon": ...}).
    """

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class BaseTask(ABC):
    @abstractmethod
    def step(self, model: nn.Module, batch) -> StepOutput: ...


def near_surface_weight(gt: torch.Tensor, lam: float, beta: float) -> torch.Tensor:
    """Near-surface weight w = 1 + λ·exp(−β·|gt|).

    Far field keeps a floor w→1 (a standard-MSE directional anchor, avoiding the
    clamp-style zero gradient -> collapse / blown-up output); the near surface
    (gt≈0) is up-weighted to 1+λ, filling the hole where the MSE gradient ∝ residual
    is weakest at the surface and detail isn't learned.
    """
    return 1.0 + lam * torch.exp(-beta * gt.abs())


class WeightedSDFLoss:
    """Near-surface weighted MSE: Σ(w·se)/Σw, w=1+λ·exp(−β|gt|).

    The normalization (÷Σw) makes β=0 give a constant w everywhere -> it cancels ->
    exactly degenerates to unweighted MSE, so switching keeps the loss scale fixed
    and kl_weight need not be re-tuned.
    """

    def __init__(self, near_weight: float = 4.0, near_beta: float = 30.0) -> None:
        self.near_weight = near_weight  # λ: extra near-surface weight (β=0 also effectively disables it)
        self.near_beta = near_beta      # β: how fast the weight decays with |sdf|; 0 → unweighted

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        w = near_surface_weight(gt, self.near_weight, self.near_beta)
        return (w * (pred - gt) ** 2).sum() / w.sum()


class ClampedSDFLoss:
    """TSDF clamp MSE (DeepSDF style): clamp pred/gt to ±δ then take MSE.

    ⚠️ Handoff note: from-scratch this causes posterior collapse (the far-field
    anchor is flattened -> an unbounded constant field can minimize the loss -> no
    latent needed). Kept only for comparison experiments, not as the main loss.
    """

    def __init__(self, clamp_val: float = 0.1) -> None:
        self.clamp_val = clamp_val

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(-self.clamp_val, self.clamp_val)
        gt = gt.clamp(-self.clamp_val, self.clamp_val)
        return F.mse_loss(pred, gt)


class SignAwareSDFLoss:
    """Weighted MSE (distance) + α·BCE (sign), decoupling the sign into a classification.

    Problem: the MSE gradient ∝ residual -> vanishes at the surface, so near-surface
    sign-acc won't move. BCE treats the same pred as an occupancy logit (inside=sdf<0);
    the classification gradient is strongest at the surface (decision boundary),
    restoring a non-vanishing gradient. sign_weight=0 → degenerates to WeightedSDFLoss.

    - sign_weight (α): weight of the sign term relative to the distance term.
    - sign_temp (k): sigmoid steepness; effective sign-gradient band ~3/k; match the
      near-surface scale (k≈30~100).

    Stability: the BCE gradient w.r.t. pred can reach ±k, far larger than MSE (~0.01);
    at init the uncalibrated far-field pred lets far points dominate the gradient and
    training bounces. So clamp the logit to ±_LOGIT_CLIP -> zero gradient outside ->
    BCE only acts on the near-surface band (far-field sign is left to MSE). Near the
    surface |logit|≲k·0.05≪clip is unaffected.
    """

    # ponytail: fixed clamp, covers the near band for k≤100 (100×0.05=5<8); widen if k reaches hundreds.
    _LOGIT_CLIP = 8.0

    def __init__(self, near_weight: float = 4.0, near_beta: float = 100.0,
                 sign_weight: float = 0.1, sign_temp: float = 30.0) -> None:
        self.near_weight = near_weight
        self.near_beta = near_beta
        self.sign_weight = sign_weight
        self.sign_temp = sign_temp

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        w = near_surface_weight(gt, self.near_weight, self.near_beta)
        mse = (w * (pred - gt) ** 2).sum() / w.sum()
        # inside=sdf<0=occupancy 1; logit=−k·pred (more negative pred -> more inside). Clamp the
        # far-field logit to stop its huge gradient dominating / bouncing training. BCE adds no w;
        # the clamp + boundary property already focus it near the surface.
        logit = (-self.sign_temp * pred).clamp(-self._LOGIT_CLIP, self._LOGIT_CLIP)
        bce = F.binary_cross_entropy_with_logits(logit, (gt < 0).to(pred.dtype))
        return mse + self.sign_weight * bce


class SignHingeSDFLoss:
    """Weighted MSE (distance) + α·hinge (sign). Penalizes the sign with basic math,
    replacing BCE.

    hinge = relu(margin − pred·sign(gt)). Correct sign (past the margin) → 0; wrong →
    linear penalty with gradient = −sign(gt), magnitude always 1 (unlike raw pred·gt,
    which |gt| zeroes out near the surface), and it does not reward large |pred|
    (unlike BCE, which drives logit→∞ and blows pred up). So it neither vanishes near
    the surface nor is inherently unstable. sign_weight=0 → degenerates to WeightedSDFLoss.

    - sign_weight (α): weight of the sign term (hinge doesn't blow up, so it can go larger than BCE).
    - sign_margin: how far pred must cross zero to count as "correct enough"; 0 = pure sign correctness.
    """

    def __init__(self, near_weight: float = 4.0, near_beta: float = 100.0,
                 sign_weight: float = 0.1, sign_margin: float = 0.0) -> None:
        self.near_weight = near_weight
        self.near_beta = near_beta
        self.sign_weight = sign_weight
        self.sign_margin = sign_margin

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        w = near_surface_weight(gt, self.near_weight, self.near_beta)
        mse = (w * (pred - gt) ** 2).sum() / w.sum()
        hinge = F.relu(self.sign_margin - pred * torch.sign(gt)).mean()
        return mse + self.sign_weight * hinge


class VAETask(BaseTask):
    """Shape VAE training task.

    Wires the encode side: `batch["query"]`/`batch["data"]` come from
    `Preprocessor` (Fourier-embedded surface points). `model.encode` returns the
    sampled latent `z` (B, num_latents, latent_dim) and per-sample `kl` (B,).

    The reconstruction loss is a swappable object (`recon_loss(pred, gt)->scalar`,
    e.g. `WeightedSDFLoss` / `ClampedSDFLoss`), Hydra-selected. `kl_weight` is the
    `r` in `recon + r * KL` and is config-driven.
    """

    def __init__(self, recon_loss, kl_weight: float = 1.0e-3, deterministic: bool = False) -> None:
        self.recon_loss = recon_loss
        self.kl_weight = kl_weight
        self.deterministic = deterministic  # True → train on z=μ (no sampling), the pure-capacity overfit path

    def step(self, model: nn.Module, batch) -> StepOutput:
        z, kl = model.encode(batch["query"], batch["data"], sample_posterior=not self.deterministic)

        return self.compute_loss(model, batch, z, kl)

    def compute_loss(self, model: nn.Module, batch, z, kl) -> StepOutput:
        pred_sdf = model.decode(z, batch["query_points"])
        recon_loss = self.recon_loss(pred_sdf, batch["gt_sdf"])

        loss = recon_loss + self.kl_weight * kl.mean()

        gt_sdf = batch["gt_sdf"]
        return StepOutput(
            loss=loss,
            metrics={
                "recon": recon_loss.item(),
                "kl": kl.mean().item(),
                # sign-IoU on the query points, as percentages; near-free (no MC).
                "viou": _VIOU(pred_sdf, gt_sdf) * 100,
                "siou": _SIOU(pred_sdf, gt_sdf) * 100,
            }
        )


if __name__ == "__main__":
    lam = 4.0
    loss = WeightedSDFLoss(near_weight=lam, near_beta=30.0)

    # self-check: for equal-size residuals, the near-surface gradient should be scaled ~(1+λ)x (the point of weighting)
    gt = torch.tensor([0.0, 0.5])                        # near surface / far field
    pred = torch.tensor([0.1, 0.6], requires_grad=True)  # both residuals = 0.1
    loss(pred, gt).backward()
    ratio = pred.grad[0].abs() / pred.grad[1].abs()
    assert abs(ratio - (1 + lam)) < 0.1, f"near/far gradient ratio {ratio:.2f} should ≈ {1+lam}"
    print(f"OK: near-surface gradient is {ratio:.2f}x the far field (=1+λ), far-field floor w=1")

    # self-check: β=0 → constant w everywhere → exactly degenerates to plain unweighted MSE (for config switching)
    gt, pred = torch.randn(64), torch.randn(64)
    unweighted = WeightedSDFLoss(near_weight=lam, near_beta=0.0)(pred, gt)
    assert torch.allclose(unweighted, ((pred - gt) ** 2).mean()), unweighted
    print("OK: β=0 equals unweighted MSE")

    # self-check: ClampedSDFLoss really clamps |val|>δ (far-field residual squashed to 0)
    clamp = ClampedSDFLoss(clamp_val=0.1)
    gt = torch.tensor([0.5])            # far field, clamps to 0.1
    pred = torch.tensor([0.9])          # clamps to 0.1 → error 0
    assert clamp(pred, gt).item() == 0.0, clamp(pred, gt)
    print("OK: ClampedSDFLoss clamps away the far-field residual")

    # self-check: on a surface sign error, SignAware's gradient (from BCE) should be far larger than plain weighted MSE
    gt = torch.tensor([1e-3])                     # just outside (target=0)
    p_w = torch.tensor([-1e-3], requires_grad=True)  # sign error: predicted inside
    p_s = p_w.detach().clone().requires_grad_(True)
    WeightedSDFLoss(near_beta=100.0)(p_w, gt).backward()
    SignAwareSDFLoss(near_beta=100.0, sign_weight=0.1, sign_temp=30.0)(p_s, gt).backward()
    assert p_s.grad.abs() > 50 * p_w.grad.abs(), (p_s.grad, p_w.grad)
    print(f"OK: SignAware surface gradient {p_s.grad.abs().item():.4f} ≫ weighted-MSE {p_w.grad.abs().item():.2e}")

    # self-check: sign direction is not flipped -- pred=0, gt<0 (inside) => ∂/∂pred>0 (pushes pred negative)
    gt = torch.tensor([-1e-3])                    # inside, target=1
    p = torch.tensor([0.0], requires_grad=True)
    SignAwareSDFLoss(sign_weight=0.1, sign_temp=30.0)(p, gt).backward()
    assert p.grad.item() > 0, p.grad
    print("OK: sign direction correct (inside point pushed negative)")

    # self-check: sign_weight=0 → degenerates to WeightedSDFLoss (one-switch off, for comparison)
    gt, pred = torch.randn(64), torch.randn(64)
    off = SignAwareSDFLoss(near_beta=100.0, sign_weight=0.0)(pred, gt)
    assert torch.allclose(off, WeightedSDFLoss(near_beta=100.0)(pred, gt)), off
    print("OK: sign_weight=0 equals plain WeightedSDFLoss")

    # self-check: at a far-field saturated point (|k·pred|≫clip) the BCE gradient is clamped to 0 → only MSE gradient (the stability fix)
    gt = torch.tensor([0.5])                        # far field
    p_w = torch.tensor([10.0], requires_grad=True)  # init-style large pred, logit=−300→clamped to −8
    p_s = p_w.detach().clone().requires_grad_(True)
    WeightedSDFLoss(near_beta=100.0)(p_w, gt).backward()
    SignAwareSDFLoss(near_beta=100.0, sign_weight=0.1, sign_temp=30.0)(p_s, gt).backward()
    assert torch.allclose(p_s.grad, p_w.grad), (p_s.grad, p_w.grad)
    print("OK: far-field saturated BCE gradient clamped away (no longer dominates / bounces)")

    # === SignHingeSDFLoss ===
    # self-check: on a near-surface sign error, hinge's contribution to pred = α (does not vanish with |gt|→0); contrast raw pred·gt which vanishes
    gt = torch.tensor([1e-3])                        # near surface, outside (want pred>0)
    p_h = torch.tensor([-0.5], requires_grad=True)   # sign error
    p_m = p_h.detach().clone().requires_grad_(True)
    SignHingeSDFLoss(near_beta=100.0, sign_weight=0.1)(p_h, gt).backward()
    WeightedSDFLoss(near_beta=100.0)(p_m, gt).backward()
    hinge_contrib = (p_h.grad - p_m.grad).abs()      # isolate the hinge term
    raw_grad = 0.1 * gt.abs()                         # gradient of raw α·relu(−pred·gt) = α·|gt|
    assert abs(hinge_contrib.item() - 0.1) < 1e-6, hinge_contrib     # = α·|sign(gt)| = 0.1
    assert hinge_contrib > 50 * raw_grad, (hinge_contrib, raw_grad)  # ≫ raw's α·1e-3
    print(f"OK: hinge near-surface sign contribution {hinge_contrib.item():.3f} (=α), raw pred·gt only {raw_grad.item():.1e}")

    # self-check: for an already-correct sign, hinge gradient=0 → no reward for large |pred|, no blow-up (BCE's illness)
    gt = torch.tensor([0.5]); p = torch.tensor([2.0], requires_grad=True)  # same sign, pred already large
    SignHingeSDFLoss(near_beta=100.0, sign_weight=0.1)(p, gt).backward()
    mse_only = WeightedSDFLoss(near_beta=100.0)
    p2 = torch.tensor([2.0], requires_grad=True); mse_only(p2, gt).backward()
    assert torch.allclose(p.grad, p2.grad), (p.grad, p2.grad)      # hinge contributes 0
    print("OK: correct sign → hinge gradient 0, no |pred| blow-up (vs BCE)")

    # self-check: sign_weight=0 → degenerates to WeightedSDFLoss
    gt, pred = torch.randn(64), torch.randn(64)
    off = SignHingeSDFLoss(near_beta=100.0, sign_weight=0.0)(pred, gt)
    assert torch.allclose(off, WeightedSDFLoss(near_beta=100.0)(pred, gt)), off
    print("OK: hinge sign_weight=0 equals plain WeightedSDFLoss")
