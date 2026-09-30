from __future__ import annotations

import torch


class VIOU:
    def __init__(self, per_shape: bool = False) -> None:
        self.per_shape = per_shape

    def _masked_iou(self, pred_in: torch.Tensor, gt_in: torch.Tensor, valid: torch.Tensor) -> float:
        if not self.per_shape:
            inter = (pred_in & gt_in & valid).sum().float()
            union = ((pred_in | gt_in) & valid).sum().float()
            return (inter / union.clamp(min=1)).item() if union > 0 else float("nan")

        flat = lambda t: t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
        pi, gi, vm = flat(pred_in), flat(gt_in), flat(valid)
        inter = (pi & gi & vm).sum(1).float()
        union = ((pi | gi) & vm).sum(1).float()
        iou = torch.where(union > 0, inter / union.clamp(min=1),
                          torch.full_like(inter, float("nan")))
        ok = ~torch.isnan(iou)
        return iou[ok].mean().item() if ok.any() else float("nan")

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        if pred_sdf.numel() == 0:
            return float("nan")
        return self._masked_iou(pred_sdf < 0, gt_sdf < 0, torch.ones_like(pred_sdf, dtype=torch.bool))


class SIOU(VIOU):
    def __init__(self, surface_tau: float = 0.02, per_shape: bool = False) -> None:
        super().__init__(per_shape=per_shape)
        self.surface_tau = surface_tau

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        # The band is passed as a mask rather than used to index: indexing flattens the batch, which
        # would make per_shape impossible (each shape has a different number of in-band points).
        m = gt_sdf.abs() < self.surface_tau
        if not m.any():
            return float("nan")
        return self._masked_iou(pred_sdf < 0, gt_sdf < 0, m)

