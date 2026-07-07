from __future__ import annotations

import torch


class OccupancyIoU:
    """Occupancy IoU (standard metric): treat sdf<0 as inside, compute
    intersection/union of the pred/gt inside sets.

    Operates on the SDF values at query points; returns a ratio in [0,1]
    (caller multiplies by 100 for a percentage).
    """

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        pi, gi = pred_sdf < 0, gt_sdf < 0
        inter = (pi & gi).sum().float()
        union = (pi | gi).sum().float()
        return (inter / union.clamp(min=1)).item()


if __name__ == "__main__":
    a = torch.tensor([-1.0, 0.5, -0.2, 0.3, -0.7])
    assert OccupancyIoU()(a, a) == 1.0, "identical -> IoU 1"
    assert OccupancyIoU()(a, -a) == 0.0, "all signs flipped -> IoU 0"
    # partial: pred inside {0,2,4}, gt inside {0,2} -> inter 2, union 3
    p = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0])
    g = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0])
    assert abs(OccupancyIoU()(p, g) - 2 / 3) < 1e-6, OccupancyIoU()(p, g)
    print("OK: OccupancyIoU")
