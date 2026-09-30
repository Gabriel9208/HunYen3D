from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _unit(n: np.ndarray) -> np.ndarray:
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


class NormalConsistency:
    def __call__(self, pred_points, pred_normals, gt_points, gt_normals) -> float:
        p, g = np.asarray(pred_points, np.float64), np.asarray(gt_points, np.float64)
        if len(p) == 0 or len(g) == 0:
            return float("nan")
        pn, gn = _unit(np.asarray(pred_normals, np.float64)), _unit(np.asarray(gt_normals, np.float64))
        _, i_pg = cKDTree(g).query(p)     # nearest gt for each pred
        _, i_gp = cKDTree(p).query(g)     # nearest pred for each gt
        nc_pg = np.abs((pn * gn[i_pg]).sum(1))
        nc_gp = np.abs((gn * pn[i_gp]).sum(1))
        return float((nc_pg.mean() + nc_gp.mean()) / 2)
