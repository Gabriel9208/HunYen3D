from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist


from src.metrics.iou import SIOU, VIOU
from src.model.shape.diffusion.flow_matching import FlowMatching

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


def chamfer_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Differentiable bidirectional Chamfer (mean nearest-neighbour Euclidean distance).

    pred (B,P,3), gt (B,G,3) -> scalar. Same ‖·‖ as the numpy metric in src/metrics/chamfer.py,
    but built from torch.cdist + min so it can be backpropped (cKDTree can't). For anchors this is
    a tiny (B,1024,1024) cdist.
    """
    d = torch.cdist(pred, gt)  # (B,P,G) pairwise L2
    return d.min(2).values.mean() + d.min(1).values.mean()


def near_surface_weight(gt: torch.Tensor, lam: float, beta: float) -> torch.Tensor:
    return 1.0 + lam * torch.exp(-beta * gt.abs())


class WeightedSDFLoss:
    def __init__(self, near_weight: float = 4.0, near_beta: float = 30.0, eps: float = 0.0,
                 exp: float = 2.0) -> None:
        self.near_weight = near_weight  # λ: extra near-surface weight (β=0 also effectively disables it)
        self.near_beta = near_beta      # β: how fast the weight decays with |sdf|; 0 → unweighted
        self.eps = eps      # Charbonnier smoothing, REQUIRED once exp<1: d|e|^p/de = p·|e|^(p-1)
                            # diverges as e→0, so a well-fitted point would get an infinite gradient.
                            # Free at exp=2, where it only adds a constant and leaves the gradient alone.
        self.exp = exp      # default exponent, for the tasks that call this with two arguments.
                            # MRLVAETask passes its own per-m exp and overrides this.

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor, exp: float | None = None) -> torch.Tensor:
        exp = self.exp if exp is None else exp
        w = near_surface_weight(gt, self.near_weight, self.near_beta)
        # (e² + ε²)^(exp/2) covers the whole family in one expression and is smooth at e=0; using e²
        # rather than |e|^exp also keeps a fractional exp from hitting a negative base (-> NaN).
        err = ((pred - gt) ** 2 + self.eps ** 2) ** (exp / 2)
        # ^(2/exp) puts every exponent back on the exp=2 magnitude. Without it mean(|e|^0.5) ≈ 0.055
        # against mean(e²) ≈ 1.5e-5 -- a 3700x swing between steps, which would both wreck Adam's
        # second-moment estimate and silently retune kl_weight. It is the identity at exp=2.
        return ((w * err).sum() / w.sum()) ** (2.0 / exp)

def SIGReg(x, num_slices=256):
    # slice sampling -- synced across devices --
    dev = dict(device=x.device)
    proj_shape = (x.size(1), num_slices)
    A = torch.randn(proj_shape, **dev)
    A /= A.norm(p=2, dim=0)

    # -- Epps-Pulley stat. see Sec. 4.3 for alt. --
    # integration points
    t = torch.linspace(-5, 5, 17, **dev)
    # theoretical CF for N(0, 1) and Gauss. window
    exp_f = torch.exp(-0.5 * t**2)

    # empirical CF -- gathered across devices --
    x_t = (x @ A).unsqueeze(2) * t  # (N, M, T)
    ecf = (1j * x_t).exp().mean(0)

    # all_reduce across distributed processes
    if dist.is_initialized():
        dist.all_reduce(ecf, op=dist.ReduceOp.AVG)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    # weighted L2 distance
    err = (ecf - exp_f).abs().square().mul(exp_f)
    N = x.size(0) * world_size

    # numerical integration over t
    # Note: torch.trapezoid is the newer alias for torch.trapz in modern PyTorch
    trapz_fn = getattr(torch, "trapezoid", torch.trapz)
    T = trapz_fn(err, t, dim=1) 

    return T.mean()


class VAETask(BaseTask):
    def __init__(self, recon_loss, kl_weight: float = 1.0e-3, deterministic: bool = False,
                 anchor_chamfer_weight: float = 0.0, anchor_mse_weight: float = 0.0,  sig_weight: float = 1.0e-3, enable_sig=False) -> None:
        self.recon_loss = recon_loss
        self.kl_weight = kl_weight
        self.sig_weight = sig_weight
        self.deterministic = deterministic  # True → train on z=μ (no sampling), the pure-capacity overfit path
        # Anchor supervision (AnchorVAE only; both 0 → base VAE / non-anchor runs unchanged). Anchors are
        # supervised toward the FPS query coords: chamfer (permutation-invariant) + index-aligned MSE.
        self.anchor_chamfer_weight = anchor_chamfer_weight
        self.anchor_mse_weight = anchor_mse_weight
        self.enable_sig = enable_sig

    def step(self, model: nn.Module, batch) -> StepOutput:
        mu = None
        if "q_xyz" in batch:   # DoubleStreamVAE: coord and normal arrive as separate streams
            z, kl = model.encode(batch["q_xyz"], batch["d_xyz"], batch["q_emb"], batch["d_emb"], sample_posterior=not self.deterministic)
        
        else:
            out = model.encode(batch["query"], batch["data"], sample_posterior=not self.deterministic, return_mean=self.enable_sig)

            if self.enable_sig: 
                z, kl, mu = out 
            else: 
                z, kl = out

        return self.compute_loss(model, batch, z, kl, mu)

    def compute_loss(self, model: nn.Module, batch, z, kl, mu=None) -> StepOutput:
        pred_sdf, anchors = model.decode(z, batch["query_points"])
        recon_loss = self.recon_loss(pred_sdf, batch["gt_sdf"])

        sig = None
        if self.enable_sig:
            sig = SIGReg(mu.reshape(-1, mu.size(-1)))
            loss = recon_loss + self.sig_weight * sig + self.kl_weight * kl.mean()
        else:
            loss = recon_loss + self.kl_weight * kl.mean()

        gt_sdf = batch["gt_sdf"]
        metrics = {
            "recon": recon_loss.item(),
            "kl": kl.mean().item(),
            "z_std": z.detach().std().item(),
            "viou": _VIOU(pred_sdf, gt_sdf) * 100,
            "siou": _SIOU(pred_sdf, gt_sdf) * 100,
        }
        if sig is not None:
            metrics["sigreg"] = sig.item()                            # is the statistic actually falling?
            metrics["sigreg_term"] = self.sig_weight * sig.item()     # compare against recon: aim for 2-10%

        if anchors is not None and (self.anchor_chamfer_weight or self.anchor_mse_weight):
            qx = batch["query_xyz"]
            cd = chamfer_l1(anchors, qx)
            mse_a = F.mse_loss(anchors, qx)
            loss = loss + self.anchor_chamfer_weight * cd + self.anchor_mse_weight * mse_a
            metrics["anchor_cd"] = cd.item()
            metrics["anchor_mse"] = mse_a.item()

        return StepOutput(loss=loss, metrics=metrics)

class MRLVAETask(VAETask):
    def __init__(self, recon_loss, m_values: list[int], m_weights: list[float], kl_weight: float = 1.0e-3,
                 deterministic: bool = False,
                 anchor_chamfer_weight: float = 0.0, anchor_mse_weight: float = 0.0,
                 m_exponents: list[float] | None = None) -> None:
        super().__init__(recon_loss, kl_weight, deterministic, anchor_chamfer_weight, anchor_mse_weight)
        if len(m_values) != len(m_weights):
            raise ValueError(f"m_values ({len(m_values)}) and m_weights ({len(m_weights)}) must be the same length")
        if m_exponents is not None and len(m_exponents) != len(m_values):
            raise ValueError(f"m_exponents ({len(m_exponents)}) and m_values ({len(m_values)}) must be the same length")
        # One recon exponent per m. A short prefix only ever carries gross shape, so squaring its
        # residual is the right ask; the full-width code is the one expected to resolve detail, and
        # exp<1 there stops the conditional mean from averaging small features away. None = every m
        # uses recon_loss's own default, i.e. the pre-existing behaviour.
        self.m_exponents = dict(zip(m_values, m_exponents)) if m_exponents is not None else None
        self.m_values = m_values      # matryoshka truncation lengths
        # Weights become the sampling distribution over m rather than a weighted sum of per-m losses:
        # one decode per step instead of len(m_values), same expected gradient, more variance.
        self.m_weights = torch.tensor(m_weights, dtype=torch.float)

    def step(self, model: nn.Module, batch) -> StepOutput:
        if not torch.is_grad_enabled():
            # Validation: always the full latent, so val/* selects best.pt on a fixed quantity and
            # stays directly comparable to a non-MRL run. The truncation curve is measured offline.
            m = max(self.m_values)
        else:
            m = self.m_values[int(torch.multinomial(self.m_weights, 1))]

        z, kl = model.encode(batch["query"], batch["data"], m, sample_posterior=not self.deterministic)
        out = self.compute_loss(model, batch, z, kl, m)
        out.metrics["m"] = m   # train/viou is measured at whichever m was drawn -- log it to read the curve
        if self.m_exponents is not None:
            out.metrics["exp"] = self.m_exponents[m]
        return out

    def compute_loss(self, model: nn.Module, batch, z, kl, m) -> StepOutput:
        pred_sdf, anchors = model.decode(z, batch["query_points"], m)
        # Only forwarded when configured, so the other loss classes (clamp / sign_aware / sign_hinge),
        # which take no exp, still work under MRL. Validation runs at max(m_values), so val/recon is
        # reported at THAT m's exponent; best_metric_key is val/viou, which is unaffected.
        kw = {"exp": self.m_exponents[m]} if self.m_exponents is not None else {}
        recon_loss = self.recon_loss(pred_sdf, batch["gt_sdf"], **kw)

        loss = recon_loss + self.kl_weight * kl.mean()

        gt_sdf = batch["gt_sdf"]
        metrics = {
            "recon": recon_loss.item(),
            "kl": kl.mean().item(),
            # sign-IoU on the query points, as percentages; near-free (no MC).
            "viou": _VIOU(pred_sdf, gt_sdf) * 100,
            "siou": _SIOU(pred_sdf, gt_sdf) * 100,
        }

        if anchors is not None and (self.anchor_chamfer_weight or self.anchor_mse_weight):
            qx = batch["query_xyz"]
            cd = chamfer_l1(anchors, qx)
            mse_a = F.mse_loss(anchors, qx)
            loss = loss + self.anchor_chamfer_weight * cd + self.anchor_mse_weight * mse_a
            metrics["anchor_cd"] = cd.item()
            metrics["anchor_mse"] = mse_a.item()

        return StepOutput(loss=loss, metrics=metrics)

        
class FlowMatchingTask(BaseTask):
    def __init__(self, 
                 flow: FlowMatching, 
                 vae: nn.Module, 
                 self_cond: bool, 
                 eval_t: tuple, 
                 vae_ckpt: str | None = None, 
                 cond_dropout: float = 0.1,
                 scaling_factor: float = 1.0) -> None:
        self.flow = flow
        self.cond_dropout = cond_dropout
        self.self_cond = self_cond
        self.eval_t = eval_t  # fixed t levels for the val z*-decode IoU metric
        self.scaling_factor = scaling_factor
        if self_cond:
            assert hasattr(vae, "anchor"), "self_cond=True needs an AnchorVAE (vae.anchor missing)"
        if vae_ckpt is not None:  
            ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            vae.load_state_dict(ckpt["model"])
        vae.eval().requires_grad_(False)  
        self.vae = vae

    def _encode(self, batch, sample_posterior: bool):
        if "q_xyz" in batch:  # DoubleStreamVAE/FrameVAE: coord and normal arrive as separate streams
            return self.vae.encode(batch["q_xyz"], batch["d_xyz"], batch["q_emb"], batch["d_emb"], sample_posterior=sample_posterior)
        return self.vae.encode(batch["query"], batch["data"], sample_posterior=sample_posterior)

    def step(self, model: nn.Module, batch) -> StepOutput:
        device = batch["q_xyz"].device if "q_xyz" in batch else batch["query"].device
        self.vae.to(device)
        with torch.no_grad():
            z, _ = self._encode(batch, sample_posterior=True)

        cond = batch["cond"]
        if torch.is_grad_enabled() and torch.rand(()) < self.cond_dropout:
            cond = torch.zeros_like(cond)
        anchor_head = self.vae.anchor if self.self_cond else None
        loss = self.flow.transport(model, z, cond, anchor_head=anchor_head, self_cond=self.self_cond, scaling_factor=self.scaling_factor)

        metrics = {"fm": loss.item()}
        if not torch.is_grad_enabled():  # validation only: decode z* -> IoU (train skips the decode cost)
            metrics.update(self._eval_iou(model, batch))
        return StepOutput(loss=loss, metrics=metrics)

    @torch.no_grad()
    def _eval_iou(self, model: nn.Module, batch) -> dict[str, float]:
        # Full ODE denoising instead of one-step estimates.
        mu, _ = self._encode(batch, sample_posterior=False)
        cond, qp, gt = batch["cond"], batch["query_points"], batch["gt_sdf"]
        g = torch.Generator(device=mu.device).manual_seed(0)
        x = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)

        steps = 50
        dt = 1.0 / steps
        anchor = None

        for i in range(steps):
            t = torch.full((mu.shape[0],), i * dt, device=mu.device)
            v, _ = model(t, cond, x, anchor=anchor)
            if self.self_cond:
                z_star = x + (1 - t[:, None, None]) * v
                anchor = self.vae.anchor(z_star)
            x = x + v * dt

        x = x / self.scaling_factor

        pred_sdf, _ = self.vae.decode(x, qp)
        
        metrics = {
            "viou_full_ode": _VIOU(pred_sdf, gt) * 100,
            "siou_full_ode": _SIOU(pred_sdf, gt) * 100,
        }
        return metrics

class SRA2FMTask(BaseTask):
    def __init__(self, 
        sra2fm: FlowMatching, 
        vae: nn.Module, 
        sra2: nn.Module,
        self_cond: bool, 
        eval_t: tuple, 
        vae_ckpt: str | None = None, 
        cond_dropout: float = 0.1,
        probe_layer: int = 0,
        lambda_sra2: float = 1.0,
        scaling_factor: float = 1.0
    ) -> None:
        
        self.sra2fm = sra2fm
        self.sra2 = sra2
        self.lambda_sra2 = lambda_sra2
        self.cond_dropout = cond_dropout
        self.self_cond = self_cond
        self.eval_t = eval_t  # fixed t levels for the val z*-decode IoU metric
        self.probe_layer = probe_layer
        self.scaling_factor = scaling_factor
        if self_cond:
            assert hasattr(vae, "anchor"), "self_cond=True needs an AnchorVAE (vae.anchor missing)"
        if vae_ckpt is not None:  
            ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=False)
            vae.load_state_dict(ckpt["model"])
        vae.eval().requires_grad_(False)  
        self.vae = vae

    def step(self, model: nn.Module, batch) -> StepOutput:
        self.vae.to(batch["query"].device)
        self.sra2.to(batch["query"].device)
        with torch.no_grad():
            z, _ = self.vae.encode(batch["query"], batch["data"], sample_posterior=True) 
        
        cond = batch["cond"]
        if torch.is_grad_enabled() and torch.rand(()) < self.cond_dropout:
            cond = torch.zeros_like(cond)         
        anchor_head = self.vae.anchor if self.self_cond else None
        loss, sra_loss, noise_pred_loss = self.sra2fm.transport(
                                              model,
                                              self.sra2,
                                              z,
                                              cond,
                                              self.probe_layer,
                                              self_cond=self.self_cond,
                                              lambda_sra2=self.lambda_sra2,
                                              scaling_factor=self.scaling_factor
                                          )

        metrics = {"fm": loss.item(), "sra_loss": sra_loss.item(), "noise_pred_loss": noise_pred_loss.item()}
        if not torch.is_grad_enabled():  # validation only: decode z* -> IoU (train skips the decode cost)
            metrics.update(self._eval_iou(model, batch))
        return StepOutput(loss=loss, metrics=metrics)

    @torch.no_grad()
    def _eval_iou(self, model: nn.Module, batch) -> dict[str, float]:
        # Full ODE denoising instead of one-step estimates.
        mu, _ = self.vae.encode(batch["query"], batch["data"], sample_posterior=False)
        cond, qp, gt = batch["cond"], batch["query_points"], batch["gt_sdf"]
        g = torch.Generator(device=mu.device).manual_seed(0)
        x = torch.randn(mu.shape, generator=g, device=mu.device, dtype=mu.dtype)

        steps = 50
        dt = 1.0 / steps
        anchor = None

        for i in range(steps):
            t = torch.full((mu.shape[0],), i * dt, device=mu.device)
            v, _ = model(t, cond, x, anchor=anchor)
            if self.self_cond:
                z_star = x + (1 - t[:, None, None]) * v
                anchor = self.vae.anchor(z_star)
            x = x + v * dt
        
        x = x / self.scaling_factor

        pred_sdf, _ = self.vae.decode(x, qp)
        
        metrics = {
            "viou_full_ode": _VIOU(pred_sdf, gt) * 100,
            "siou_full_ode": _SIOU(pred_sdf, gt) * 100,
        }
        return metrics


if __name__ == "__main__":
    # The two hazards of a per-m exponent: exp=2 must stay bit-identical to the old MSE, and exp<1
    # must not produce an infinite gradient where the fit is already exact.
    torch.manual_seed(0)
    gt = torch.randn(2, 4096, 1) * 0.15
    pred = gt + torch.randn_like(gt) * 0.0039
    w = near_surface_weight(gt, 4.0, 30.0)
    assert torch.equal(WeightedSDFLoss(4.0, 30.0)(pred, gt),
                       (w * (pred - gt) ** 2).sum() / w.sum())
    for eps, finite in ((0.0, False), (1e-3, True)):
        z = gt[:1, :1].clone()
        q = z.clone().requires_grad_(True)          # residual exactly 0, the worst case for exp<1
        WeightedSDFLoss(4.0, 30.0, eps=eps)(q, z, exp=0.5).backward()
        assert torch.isfinite(q.grad).all().item() is finite, (eps, q.grad)
    # the ^(2/exp) rescale must keep every exponent within an order of magnitude of exp=2
    mags = [WeightedSDFLoss(4.0, 30.0, eps=1e-3)(pred, gt, exp=e).item() for e in (2.0, 1.5, 1.0, 0.5)]
    assert max(mags) / min(mags) < 10, mags
    print("OK: WeightedSDFLoss exponent")

