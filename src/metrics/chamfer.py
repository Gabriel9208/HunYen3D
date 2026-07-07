from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class ChamferDistance:
    """Bidirectional Chamfer distance (standard metric): sum of each surface
    point cloud's mean nearest-neighbour distance to the other.

    Point clouds in, scalar out -- deliberately does not touch meshes; getting
    the points (pred marching-cubes / gt surface sampling) is the caller's job.
    L1 (default): mean‖p->nearest g‖ + mean‖g->nearest p‖.
    """

    def __call__(self, pred_points, gt_points) -> float:
        p = np.asarray(pred_points, dtype=np.float64)
        g = np.asarray(gt_points, dtype=np.float64)
        d_pg, _ = cKDTree(g).query(p)     # each pred point to its nearest gt point
        d_gp, _ = cKDTree(p).query(g)     # each gt point to its nearest pred point
        return float(d_pg.mean() + d_gp.mean())


if __name__ == "__main__":
    pts = np.random.rand(500, 3)
    assert ChamferDistance()(pts, pts) == 0.0, "identical clouds -> 0"
    # single-point pair: distance 1, one each way -> 2 (known answer)
    cd = ChamferDistance()(np.array([[0.0, 0, 0]]), np.array([[1.0, 0, 0]]))
    assert cd == 2.0, cd
    # a cluster shifted far (>> internal spacing) -> both directions ≈ the shift
    a = np.random.rand(300, 3) * 0.01
    cd2 = ChamferDistance()(a, a + np.array([5.0, 0, 0]))
    assert abs(cd2 - 10.0) < 0.1, cd2
    print("OK: ChamferDistance")
