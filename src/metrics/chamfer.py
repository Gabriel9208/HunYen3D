from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class ChamferDistance:
    def __call__(self, pred_points, gt_points) -> float:
        p = np.asarray(pred_points, dtype=np.float64)
        g = np.asarray(gt_points, dtype=np.float64)
        d_pg, _ = cKDTree(g).query(p)     # pred to nearest gt 
        d_gp, _ = cKDTree(p).query(g)     # gt to nearest pred 
        return float(d_pg.mean() + d_gp.mean())

