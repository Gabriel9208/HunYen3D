from __future__ import annotations

import numpy as np
import trimesh


def sample_points(shape, n: int) -> np.ndarray | None:
    """Surface point cloud (n, 3) float32 from a mesh, or resampled from a point array.

    shape: trimesh.Trimesh (surface-sampled) | (N,3) array-like (resampled to n) | None.
    Returns None when there is no geometry to sample (empty mesh / empty array) so the
    caller can emit the float("nan") sentinel, matching the other metrics.
    """
    if shape is None:
        return None
    if isinstance(shape, trimesh.Trimesh):
        if shape.faces is None or len(shape.faces) == 0:
            return None
        pts, _ = trimesh.sample.sample_surface(shape, n)
        return np.asarray(pts, dtype=np.float32)
    pts = np.asarray(shape, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    idx = np.random.choice(pts.shape[0], n, replace=pts.shape[0] < n)
    return pts[idx]


def unit_ball_norm(pts: np.ndarray) -> np.ndarray:
    """Center to centroid, scale so the farthest point sits at radius 1.

    This is the `pc_normalize` both ULIP (data/dataset_3d.py) and Uni3D apply before
    their encoders — the encoders are trained on unit-ball clouds, so skipping it gives
    meaningless embeddings.
    """
    c = pts.mean(axis=0)
    p = pts - c
    r = np.linalg.norm(p, axis=1).max()
    return p / max(float(r), 1e-9)
