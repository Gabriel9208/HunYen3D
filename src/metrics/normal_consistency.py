from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _unit(n: np.ndarray) -> np.ndarray:
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


class NormalConsistency:
    """Normal consistency between two oriented surface point clouds (standard 3D
    reconstruction metric): for each point, take |cos| between its normal and the
    normal of its nearest neighbour in the other cloud, averaged over both directions.

    |cos| (not cos) so a globally flipped orientation convention doesn't penalise a
    geometrically correct surface. Points+normals in, scalar in [0,1] out. Empty cloud
    → NaN (uncomputable sentinel).
    """

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


if __name__ == "__main__":
    import math

    pts = np.random.rand(400, 3)
    nrm = _unit(np.random.rand(400, 3) - 0.5)
    assert abs(NormalConsistency()(pts, nrm, pts, nrm) - 1.0) < 1e-9, "same points+normals -> 1"
    # flipped normals still score 1 (|cos| is orientation-agnostic)
    assert abs(NormalConsistency()(pts, nrm, pts, -nrm) - 1.0) < 1e-9, "flipped -> still 1"
    # orthogonal normals -> 0
    ex = np.tile([1.0, 0, 0], (400, 1)); ey = np.tile([0.0, 1, 0], (400, 1))
    assert NormalConsistency()(pts, ex, pts, ey) < 1e-9, "orthogonal -> 0"
    assert math.isnan(NormalConsistency()(np.zeros((0, 3)), np.zeros((0, 3)), pts, nrm)), "empty -> NaN"
    print("OK: NormalConsistency")
