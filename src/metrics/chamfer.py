from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class ChamferDistance:
    """雙向 Chamfer 距離(標準指標):兩表面點雲各自最近鄰距離的平均和。

    點雲進、純量出——刻意不碰網格;取點(pred marching-cubes / gt 表面取樣)是呼叫端的事。
    L1(預設):mean‖p→最近g‖ + mean‖g→最近p‖。
    """

    def __call__(self, pred_points, gt_points) -> float:
        p = np.asarray(pred_points, dtype=np.float64)
        g = np.asarray(gt_points, dtype=np.float64)
        d_pg, _ = cKDTree(g).query(p)     # 每個 pred 點到最近 gt 點
        d_gp, _ = cKDTree(p).query(g)     # 每個 gt 點到最近 pred 點
        return float(d_pg.mean() + d_gp.mean())


if __name__ == "__main__":
    pts = np.random.rand(500, 3)
    assert ChamferDistance()(pts, pts) == 0.0, "相同點雲 → 0"
    # 單點對:距離 1,雙向各 1 → 2(已知答案)
    cd = ChamferDistance()(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
    assert cd == 2.0, cd
    # 一團點整體平移遠(>> 內部間距)→ 雙向都 ≈ 平移量
    a = np.random.rand(300, 3) * 0.01
    cd2 = ChamferDistance()(a, a + np.array([5.0, 0, 0]))
    assert abs(cd2 - 10.0) < 0.1, cd2
    print("OK: ChamferDistance")
