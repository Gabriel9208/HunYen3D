from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


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
    """近表面加權 w = 1 + λ·exp(−β·|gt|)。

    遠場保底 w→1(標準 MSE 定向錨,避免 clamp 那種梯度歸零→collapse/輸出爆走);
    近表面(gt≈0)加重到 1+λ,補上 MSE 梯度∝殘差在表面最弱、學不到細節的洞。
    """
    return 1.0 + lam * torch.exp(-beta * gt.abs())


class WeightedSDFLoss:
    """近表面加權 MSE:Σ(w·se)/Σw,w=1+λ·exp(−β|gt|)。

    正規化(÷Σw)讓 β=0 時 w 處處常數→上下約掉→精準退化成 unweighted MSE,
    切換時 loss 尺度不變、不用重調 kl_weight。
    """

    def __init__(self, near_weight: float = 4.0, near_beta: float = 30.0) -> None:
        self.near_weight = near_weight  # λ:近表面額外權重(β=0 亦等效關閉)
        self.near_beta = near_beta      # β:權重隨 |sdf| 衰減速度;0 → unweighted

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        w = near_surface_weight(gt, self.near_weight, self.near_beta)
        return (w * (pred - gt) ** 2).sum() / w.sum()


class ClampedSDFLoss:
    """TSDF clamp MSE(DeepSDF 式):pred/gt 夾 ±δ 後取 MSE。

    ⚠️ 交接記錄:from-scratch 用它會 posterior collapse(遠場錨被夾平→無界常數場
    就能壓 loss→不需 latent)。保留僅供對照實驗,勿當主力。
    """ 

    def __init__(self, clamp_val: float = 0.1) -> None:
        self.clamp_val = clamp_val

    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(-self.clamp_val, self.clamp_val)
        gt = gt.clamp(-self.clamp_val, self.clamp_val)
        return F.mse_loss(pred, gt)


class SignAwareSDFLoss:
    """加權 MSE(距離)+ α·BCE(符號),把「符號」解耦成分類。

    病灶:MSE 在表面梯度∝殘差→趨零,近表面 sign-acc 學不動。BCE 把同一個 pred
    當 occupancy logit(inside=sdf<0),分類梯度在表面(決策邊界)最強,補回非消失
    梯度。sign_weight=0 → 退化成 WeightedSDFLoss。

    - sign_weight (α):符號項相對距離項的比重。
    - sign_temp (k):sigmoid 陡度,有效符號梯度帶寬 ~3/k;對齊近表面尺度(k≈30~100)。

    穩定性:BCE 對 pred 的梯度可達 ±k,遠比 MSE(~0.01)大;init 時遠場 pred 未校準
    會讓遠場點主導梯度、訓練亂跳。故把 logit 夾在 ±_LOGIT_CLIP,夾點外梯度歸零 →
    BCE 只作用在近表面帶(遠場符號交給 MSE)。近表面 |logit|≲k·0.05≪clip 不受影響。
    """

    # ponytail: 固定夾值,涵蓋 k≤100 的近表面帶(100×0.05=5<8);k 若開到數百需放大。
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
        # inside=sdf<0=occupancy 1;logit=−k·pred(pred 越負→越 inside)。夾住遠場 logit
        # 避免其巨大梯度主導、訓練亂跳。BCE 不另加 w,夾+邊界性質已讓它聚焦近表面。
        logit = (-self.sign_temp * pred).clamp(-self._LOGIT_CLIP, self._LOGIT_CLIP)
        bce = F.binary_cross_entropy_with_logits(logit, (gt < 0).to(pred.dtype))
        return mse + self.sign_weight * bce


class SignHingeSDFLoss:
    """加權 MSE(距離)+ α·hinge(符號)。用基本數學罰符號錯,取代 BCE。

    hinge = relu(margin − pred·sign(gt))。符號對了(且超過 margin)→ 0;錯了→線性罰,
    梯度 = −sign(gt),大小恆為 1(不像 raw pred·gt 會被 |gt| 在近表面歸零),且不獎勵
    大 |pred|(不像 BCE 逼 logit→∞ 而吹爆 pred)。故近表面不消失、又天生穩定。
    sign_weight=0 → 退化成 WeightedSDFLoss。

    - sign_weight (α):符號項比重(hinge 不吹爆,可比 BCE 放大)。
    - sign_margin:要求 pred 越過零多少才算「夠對」;0 = 純符號正確。
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

    def __init__(self, recon_loss, kl_weight: float = 1.0e-3) -> None:
        self.recon_loss = recon_loss
        self.kl_weight = kl_weight

    def step(self, model: nn.Module, batch) -> StepOutput:
        z, kl = model.encode(batch["query"], batch["data"])

        return self.compute_loss(model, batch, z, kl)

    def compute_loss(self, model: nn.Module, batch, z, kl) -> StepOutput:
        pred_sdf = model.decode(z, batch["query_points"])
        recon_loss = self.recon_loss(pred_sdf, batch["gt_sdf"])

        loss = recon_loss + self.kl_weight * kl.mean()
        
        return StepOutput(
            loss=loss,
            metrics={
                "recon": recon_loss.item(),
                "kl": kl.mean().item()
            }
        )


if __name__ == "__main__":
    lam = 4.0
    loss = WeightedSDFLoss(near_weight=lam, near_beta=30.0)

    # self-check:同樣大小的殘差,近表面點的梯度應被放大 ~(1+λ) 倍(加權的目的)
    gt = torch.tensor([0.0, 0.5])                        # 近表面 / 遠場
    pred = torch.tensor([0.1, 0.6], requires_grad=True)  # 兩點殘差都 = 0.1
    loss(pred, gt).backward()
    ratio = pred.grad[0].abs() / pred.grad[1].abs()
    assert abs(ratio - (1 + lam)) < 0.1, f"近/遠梯度比 {ratio:.2f} 應 ≈ {1+lam}"
    print(f"OK: 近表面梯度是遠場的 {ratio:.2f}x (=1+λ),遠場保底 w=1")

    # self-check:β=0 → w 處處常數 → 精準退化成純 unweighted MSE(config 切換用)
    gt, pred = torch.randn(64), torch.randn(64)
    unweighted = WeightedSDFLoss(near_weight=lam, near_beta=0.0)(pred, gt)
    assert torch.allclose(unweighted, ((pred - gt) ** 2).mean()), unweighted
    print("OK: β=0 等於 unweighted MSE")

    # self-check:ClampedSDFLoss 真的把 |值|>δ 夾掉(遠場殘差被壓成 0)
    clamp = ClampedSDFLoss(clamp_val=0.1)
    gt = torch.tensor([0.5])            # 遠場,夾後 = 0.1
    pred = torch.tensor([0.9])          # 夾後 = 0.1 → 誤差 0
    assert clamp(pred, gt).item() == 0.0, clamp(pred, gt)
    print("OK: ClampedSDFLoss 夾掉遠場殘差")

    # self-check:表面符號錯時,SignAware 的梯度(BCE 補的)應遠大於純加權 MSE
    gt = torch.tensor([1e-3])                     # 剛好在外(target=0)
    p_w = torch.tensor([-1e-3], requires_grad=True)  # 符號錯:預測在內
    p_s = p_w.detach().clone().requires_grad_(True)
    WeightedSDFLoss(near_beta=100.0)(p_w, gt).backward()
    SignAwareSDFLoss(near_beta=100.0, sign_weight=0.1, sign_temp=30.0)(p_s, gt).backward()
    assert p_s.grad.abs() > 50 * p_w.grad.abs(), (p_s.grad, p_w.grad)
    print(f"OK: SignAware 表面梯度 {p_s.grad.abs().item():.4f} ≫ 加權MSE {p_w.grad.abs().item():.2e}")

    # self-check:符號方向沒接反 —— pred=0、gt<0(inside) 時 ∂/∂pred>0(逼 pred 變負)
    gt = torch.tensor([-1e-3])                    # inside,target=1
    p = torch.tensor([0.0], requires_grad=True)
    SignAwareSDFLoss(sign_weight=0.1, sign_temp=30.0)(p, gt).backward()
    assert p.grad.item() > 0, p.grad
    print("OK: 符號方向正確(inside 點被逼向負)")

    # self-check:sign_weight=0 → 退化成 WeightedSDFLoss(一鍵關掉,對照用)
    gt, pred = torch.randn(64), torch.randn(64)
    off = SignAwareSDFLoss(near_beta=100.0, sign_weight=0.0)(pred, gt)
    assert torch.allclose(off, WeightedSDFLoss(near_beta=100.0)(pred, gt)), off
    print("OK: sign_weight=0 等於純 WeightedSDFLoss")

    # self-check:遠場飽和點(|k·pred|≫clip)的 BCE 梯度被夾成 0 → 只剩 MSE 梯度(穩定性修法)
    gt = torch.tensor([0.5])                        # 遠場
    p_w = torch.tensor([10.0], requires_grad=True)  # init 式的大 pred,logit=−300→夾到 −8
    p_s = p_w.detach().clone().requires_grad_(True)
    WeightedSDFLoss(near_beta=100.0)(p_w, gt).backward()
    SignAwareSDFLoss(near_beta=100.0, sign_weight=0.1, sign_temp=30.0)(p_s, gt).backward()
    assert torch.allclose(p_s.grad, p_w.grad), (p_s.grad, p_w.grad)
    print("OK: 遠場飽和點 BCE 梯度被夾掉(不再主導、不亂跳)")

    # === SignHingeSDFLoss ===
    # self-check:近表面符號錯,hinge 對 pred 的貢獻大小=α(不隨 |gt|→0 消失);對比 raw pred·gt 會消失
    gt = torch.tensor([1e-3])                        # 貼表面、在外(want pred>0)
    p_h = torch.tensor([-0.5], requires_grad=True)   # 符號錯
    p_m = p_h.detach().clone().requires_grad_(True)
    SignHingeSDFLoss(near_beta=100.0, sign_weight=0.1)(p_h, gt).backward()
    WeightedSDFLoss(near_beta=100.0)(p_m, gt).backward()
    hinge_contrib = (p_h.grad - p_m.grad).abs()      # 隔離出 hinge 那一項
    raw_grad = 0.1 * gt.abs()                         # raw α·relu(−pred·gt) 的梯度 = α·|gt|
    assert abs(hinge_contrib.item() - 0.1) < 1e-6, hinge_contrib     # = α·|sign(gt)| = 0.1
    assert hinge_contrib > 50 * raw_grad, (hinge_contrib, raw_grad)  # ≫ raw 的 α·1e-3
    print(f"OK: hinge 近表面符號貢獻 {hinge_contrib.item():.3f} (=α),raw pred·gt 只有 {raw_grad.item():.1e}")

    # self-check:符號已對的點,hinge 梯度=0 → 不獎勵大 |pred|、不吹爆(BCE 的病)
    gt = torch.tensor([0.5]); p = torch.tensor([2.0], requires_grad=True)  # 同號、pred 已很大
    SignHingeSDFLoss(near_beta=100.0, sign_weight=0.1)(p, gt).backward()
    mse_only = WeightedSDFLoss(near_beta=100.0)
    p2 = torch.tensor([2.0], requires_grad=True); mse_only(p2, gt).backward()
    assert torch.allclose(p.grad, p2.grad), (p.grad, p2.grad)      # hinge 貢獻 0
    print("OK: 符號已對→hinge 梯度 0,不吹 |pred|(對比 BCE)")

    # self-check:sign_weight=0 → 退化成 WeightedSDFLoss
    gt, pred = torch.randn(64), torch.randn(64)
    off = SignHingeSDFLoss(near_beta=100.0, sign_weight=0.0)(pred, gt)
    assert torch.allclose(off, WeightedSDFLoss(near_beta=100.0)(pred, gt)), off
    print("OK: hinge sign_weight=0 等於純 WeightedSDFLoss")
