from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class FScore:
    """F-score@tau between two surface point clouds (standard 3D reconstruction metric).

    precision = fraction of pred points within tau of some gt point (accuracy);
    recall    = fraction of gt points within tau of some pred point (completeness);
    F = 2·P·R / (P + R).

    Point clouds in, scalar out -- mesh sampling (pred marching-cubes / gt surface) is
    the caller's job. Empty cloud → NaN (uncomputable sentinel). tau is in the same units
    as the points (here the [-1,1] normalised frame).
    """

    def __init__(self, tau: float = 0.01) -> None:
        self.tau = tau

    def __call__(self, pred_points, gt_points) -> float:
        p = np.asarray(pred_points, dtype=np.float64)
        g = np.asarray(gt_points, dtype=np.float64)
        if len(p) == 0 or len(g) == 0:
            return float("nan")
        d_pg, _ = cKDTree(g).query(p)     # each pred point to nearest gt
        d_gp, _ = cKDTree(p).query(g)     # each gt point to nearest pred
        precision = float((d_pg < self.tau).mean())
        recall = float((d_gp < self.tau).mean())
        if precision + recall == 0.0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    import math

    pts = np.random.rand(500, 3)
    assert FScore(tau=0.01)(pts, pts) == 1.0, "identical clouds -> F 1"
    # clouds shifted far apart (>> tau) -> no matches -> F 0
    a = np.random.rand(300, 3)
    assert FScore(tau=0.01)(a, a + 5.0) == 0.0, "disjoint clouds -> F 0"
    assert math.isnan(FScore()(np.zeros((0, 3)), pts)), "empty -> NaN"
    print("OK: FScore")
