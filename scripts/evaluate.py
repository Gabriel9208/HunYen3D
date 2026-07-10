"""Geometric reconstruction eval (merges the old eval_recon + viz_recon).

Answers "is an MSE of 0.0x actually good or bad?". MSE is dragged down by easy far-field points,
so we report MSE banded by |gt_sdf| plus geometrically meaningful metrics. Reconstructs with a
fixed mu (no sampling) on each shape's own query points.

Default (cheap, SDF-space only):
  - occupancy IoU, banded RMS by distance (rms_ratio vs the predict-all-0 baseline)
+viz=true (marching-cubes each shape, heavier):
  - Chamfer distance (recon surface vs gt surface)
  - GT|recon comparison PNG + .off meshes saved to results/<name>/

Usage:
  uv run python scripts/evaluate.py +experiment=base_small +ckpt=/abs/last.pt +shapes=16
  uv run python scripts/evaluate.py +experiment=base_small +ckpt=/abs/best.pt +shapes=8 \
      +viz=true [+name=recon] [+resolution=128]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.engine.utils import get_device, set_seed
from src.metrics import BandedSDFMetrics, ChamferDistance, OccupancyIoU

_CHAMFER_SAMPLES = 50_000  # ponytail: surface points per cloud; plenty for a stable estimate, still cheap


def _viz_setup(cfg: DictConfig, ds):
    """Heavy-dependency setup for +viz, imported lazily so metric-only runs stay light.

    Returns a namespace with `post`, `renderer`, `out` (results dir) and the `load_gt` /
    `save_pair` helpers (moved here from the old viz_recon.py)."""
    import gzip
    from types import SimpleNamespace

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless off-screen; before importing pyrender
    import mesh_to_sdf  # noqa: F401 — must precede pyrender (it monkeypatches OpenGL); Preprocessor uses it too
    import numpy as np
    import pyrender
    import trimesh
    from hydra.utils import get_original_cwd
    from PIL import Image

    from src.model.shape.VAE.postprocess import Postprocess

    W = H = 512

    def load_gt(mesh_path: str) -> trimesh.Trimesh:
        # gz-aware load (duplicates preprocess.build_cache; not worth refactoring for one caller).
        if mesh_path.endswith(".gz"):
            inner = os.path.splitext(mesh_path[:-3])[1].lstrip(".")
            with gzip.open(mesh_path, "rb") as f:
                mesh = trimesh.load(f, file_type=inner, force="mesh")
        else:
            mesh = trimesh.load_mesh(mesh_path, force="mesh")
        return ds.preprocessor.mesh2query.normalize_mesh(mesh, 0.98)

    def _look_at(eye, target=(0, 0, 0), up=(0, 1, 0)) -> np.ndarray:
        eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
        f = target - eye
        f /= np.linalg.norm(f)
        r = np.cross(f, up)
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = r, u, -f, eye  # camera looks down -Z
        return pose

    renderer = pyrender.OffscreenRenderer(W, H)

    def _render(mesh) -> np.ndarray:
        if mesh is None:
            return np.zeros((H, W, 3), np.uint8)  # no reconstruction → black panel
        scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[0, 0, 0])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        pose = _look_at(eye=(2.0, 1.5, 2.6))  # 3/4 view of the [-1,1] cube
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)  # light follows the camera
        color, _ = renderer.render(scene)
        return color[..., :3]

    def save_pair(base: str, gt_mesh, recon_mesh) -> None:
        os.makedirs(os.path.dirname(base), exist_ok=True)
        gt_mesh.export(base + "_gt.off")
        if recon_mesh is not None:
            recon_mesh.export(base + "_recon.off")
        cmp = np.hstack([_render(gt_mesh), _render(recon_mesh)])  # left GT | right recon
        Image.fromarray(cmp).save(base + "_cmp.png")

    out = os.path.join(get_original_cwd(), "results", cfg.get("name", "recon"))  # project root, not the hydra run dir
    post = Postprocess(ds.preprocessor.fourier_embedder, resolution=int(cfg.get("resolution", 128)))
    return SimpleNamespace(post=post, renderer=renderer, out=out, load_gt=load_gt, save_pair=save_pair)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(0)
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device, weights_only=False)["model"])

    ds = instantiate(cfg.data.val.dataset)
    n = min(int(cfg.get("shapes", 16)), len(ds))
    viz = bool(cfg.get("viz", False))
    v = _viz_setup(cfg, ds) if viz else None

    preds, gts = [], []
    chamfers: list[float] = []
    no_surface = 0
    with torch.no_grad():
        for i in range(n):
            it = ds[i]
            mu, _ = model.encoder(it["query"].unsqueeze(0).to(device), it["data"].unsqueeze(0).to(device))
            pred = model.decode(mu, it["query_points"].unsqueeze(0).to(device)).squeeze(0).cpu()
            preds.append(pred)
            gts.append(it["gt_sdf"])

            if viz:
                recon = v.post.to_mesh(model.decode, mu)
                gt_mesh = v.load_gt(ds.paths[i])
                rel = os.path.splitext(os.path.relpath(ds.paths[i], ds.obj_root).removesuffix(".gz"))[0]
                v.save_pair(os.path.join(v.out, rel), gt_mesh, recon)
                if recon is None:
                    no_surface += 1
                    print(f"[{i}] {rel}  no zero-crossing surface (None)")
                else:
                    cd = ChamferDistance()(recon.sample(_CHAMFER_SAMPLES), gt_mesh.sample(_CHAMFER_SAMPLES))
                    chamfers.append(cd)
                    print(f"[{i}] {rel}  recon verts={len(recon.vertices)}  chamfer={cd:.5f}")

    pred = torch.cat(preds).flatten()
    gt = torch.cat(gts).flatten()
    diag = BandedSDFMetrics()(pred, gt)
    iou = OccupancyIoU()(pred, gt)

    print(f"\nshapes={n}  points={gt.numel():,}  ckpt={cfg.ckpt}")
    print(f"overall MSE      : {(pred - gt).pow(2).mean():.5f}")
    print(f"overall RMS      : {(pred - gt).pow(2).mean().sqrt():.4f}  (SDF scale ~[-1,1])")
    print(f"occupancy IoU    : {iou * 100:.2f}%")
    if viz:
        mean_cd = sum(chamfers) / len(chamfers) if chamfers else float("nan")
        print(f"chamfer (mean)   : {mean_cd:.5f}  over {len(chamfers)}/{n} shapes with a surface "
              f"(no surface {no_surface}/{n})")

    print("banded RMS by |gt| (rms_ratio = band RMS / predict-all-0 RMS; <1 = beats predicting zero, ≈1 = learned nothing):")
    for name, b in diag["bands"].items():
        print(f"  {name:<16} frac{b['frac']:5.1f}%  RMS={b['rms']:.5f}  "
              f"ratio={b['rms_ratio']:5.2f}  (baseline pred0-RMS={b['pred0'] ** 0.5:.5f})")

    if viz:
        v.renderer.delete()
        print(f"\nviz → {v.out}/  (PNG left=GT right=recon)")


if __name__ == "__main__":
    main()
