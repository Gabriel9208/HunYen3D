from __future__ import annotations

import torch


class OccupancyIoU:
    """Occupancy IoU(標準指標):sdf<0 視為內部,算 pred/gt 內部集合的 交集/聯集。

    在 query point 的 SDF 值上算,回傳 [0,1] 比值(呼叫端要百分比自己 ×100)。
    """

    def __call__(self, pred_sdf: torch.Tensor, gt_sdf: torch.Tensor) -> float:
        pi, gi = pred_sdf < 0, gt_sdf < 0
        inter = (pi & gi).sum().float()
        union = (pi | gi).sum().float()
        return (inter / union.clamp(min=1)).item()


if __name__ == "__main__":
    a = torch.tensor([-1.0, 0.5, -0.2, 0.3, -0.7])
    assert OccupancyIoU()(a, a) == 1.0, "相同 → IoU 1"
    assert OccupancyIoU()(a, -a) == 0.0, "全反號 → IoU 0"
    # 半對:pred 內部 {0,2,4},gt 內部 {0,2} → 交集2 聯集3
    p = torch.tensor([-1.0, 1.0, -1.0, 1.0, -1.0])
    g = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0])
    assert abs(OccupancyIoU()(p, g) - 2 / 3) < 1e-6, OccupancyIoU()(p, g)
    print("OK: OccupancyIoU")
