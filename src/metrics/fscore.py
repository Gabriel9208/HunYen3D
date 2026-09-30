from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class FScore:
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

