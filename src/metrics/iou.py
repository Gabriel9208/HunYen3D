from __future__ import annotations

import torch


class VIOU:
    """Volume occupancy IoU (standard metric; formerly OccupancyIoU): treat sdf<0
    as inside, compute intersection/union of the pred/gt inside sets over ALL the
    given query points.

    Operates on the SDF values at query points; returns a ratio in [0,1] (caller
    multiplies by 100 for a percentage). Empty input → NaN (uncomputable sentinel).
    """

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        if pred_sdf.numel() == 0:
            return float("nan")
        pi, gi = pred_sdf < 0, gt_sdf < 0
        inter = (pi & gi).sum().float()
        union = (pi | gi).sum().float()
        return (inter / union.clamp(min=1)).item()


class SIOU(VIOU):
    """Surface occupancy IoU (standard metric): VIOU restricted to the near-surface
    band |gt| < surface_tau, so a low overall loss dominated by easy far-field points
    cannot inflate it. Self-contained — takes the full pred/gt and masks internally.
    Empty band → NaN (uncomputable sentinel).
    """

    def __init__(self, surface_tau: float = 0.02) -> None:
        self.surface_tau = surface_tau

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        m = gt_sdf.abs() < self.surface_tau
        if not m.any():
            return float("nan")
        return super().__call__(pred_sdf[m], gt_sdf[m])


if __name__ == "__main__":
    import math

    a = torch.tensor([-1.0, 0.5, -0.2, 0.3, -0.7])
    assert VIOU()(a, a) == 1.0, "identical -> IoU 1"
    assert VIOU()(a, -a) == 0.0, "all signs flipped -> IoU 0"
    # partial: pred inside {0,2,4}, gt inside {0,2} -> inter 2, union 3
    p = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0])
    g = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0])
    assert abs(VIOU()(p, g) - 2 / 3) < 1e-6, VIOU()(p, g)
    assert math.isnan(VIOU()(torch.empty(0), torch.empty(0))), "empty -> NaN"

    # SIOU restricts to |gt|<tau: only the near points (indices 0,1) count here.
    gt = torch.tensor([0.005, -0.01, 0.5, -0.5])       # two near, two far
    pred = torch.tensor([0.005, -0.01, 0.5, 0.5])      # near correct, one far sign wrong
    assert SIOU(surface_tau=0.02)(pred, gt) == 1.0, "near band all correct -> S-IoU 1"
    assert math.isnan(SIOU(surface_tau=1e-4)(gt, gt)), "no point in band -> NaN"
    # a near-band sign error drops S-IoU below 1
    pred2 = torch.tensor([0.005, 0.01, 0.5, -0.5])     # index1 sign flipped (near)
    assert SIOU(surface_tau=0.02)(pred2, gt) < 1.0, SIOU(surface_tau=0.02)(pred2, gt)
    print("OK: VIOU / SIOU")
