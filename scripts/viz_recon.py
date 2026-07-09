"""Visualize reconstruction quality: encoder → decoder → Postprocess (marching cubes).

For a few val shapes: encode to mu, decode an SDF grid, marching-cubes back to a mesh, save .off,
and render off-screen with pyrender, side by side with the normalized GT as one PNG (left GT, right recon).

Usage:
  uv run python scripts/viz_recon.py +experiment=base_small \
      +ckpt=/abs/best.pt +shapes=8 [+resolution=128] [+name=recon]
"""

from __future__ import annotations

import gzip
import os
import sys

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless off-screen; must be set before importing pyrender

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import mesh_to_sdf  # noqa: F401 — must be imported before pyrender (it monkeypatches OpenGL); Preprocessor uses it too
import numpy as np
import pyrender
import torch
import trimesh
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from PIL import Image

from src.engine.utils import get_device, set_seed
from src.model.shape.VAE.postprocess import Postprocess

W = H = 512


def load_normalized(mesh_path: str, preprocessor) -> trimesh.Trimesh:
    # ponytail: this gz-aware loading duplicates preprocess.build_cache:330-335; not worth refactoring preprocess for a single caller.
    if mesh_path.endswith(".gz"):
        inner = os.path.splitext(mesh_path[:-3])[1].lstrip(".")
        with gzip.open(mesh_path, "rb") as f:
            mesh = trimesh.load(f, file_type=inner, force="mesh")
    else:
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
    return preprocessor.mesh2query.normalize_mesh(mesh, 0.98)


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


def render(mesh: trimesh.Trimesh | None, renderer: pyrender.OffscreenRenderer) -> np.ndarray:
    if mesh is None:
        return np.zeros((H, W, 3), np.uint8)  # no reconstruction → black panel
    scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[0, 0, 0])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    pose = _look_at(eye=(2.0, 1.5, 2.6))  # 3/4 view of the [-1,1] cube
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)  # light follows the camera
    color, _ = renderer.render(scene)
    return color[..., :3]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(0)
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device, weights_only=False)["model"])

    ds = instantiate(cfg.data.val.dataset)
    post = Postprocess(ds.preprocessor.fourier_embedder, resolution=int(cfg.get("resolution", 128)))
    out = os.path.join(get_original_cwd(), "results", cfg.get("name", "recon"))  # pin to project root, not the hydra run dir
    n = min(int(cfg.get("shapes", 8)), len(ds))

    renderer = pyrender.OffscreenRenderer(W, H)
    no_surface = 0
    with torch.no_grad():
        for i in range(n):
            it = ds[i]
            mu, _ = model.encoder(it["query"].unsqueeze(0).to(device), it["data"].unsqueeze(0).to(device))
            recon = post.to_mesh(model.decode, mu)
            gt = load_normalized(ds.paths[i], ds.preprocessor)

            rel = os.path.splitext(os.path.relpath(ds.paths[i], ds.obj_root).removesuffix(".gz"))[0]
            base = os.path.join(out, rel)
            os.makedirs(os.path.dirname(base), exist_ok=True)
            gt.export(base + "_gt.off")
            if recon is None:
                no_surface += 1
                print(f"[{i}] {rel}  no zero-crossing surface (None)")
            else:
                recon.export(base + "_recon.off")
                print(f"[{i}] {rel}  recon verts={len(recon.vertices)}")

            cmp = np.hstack([render(gt, renderer), render(recon, renderer)])  # left GT | right recon
            Image.fromarray(cmp).save(base + "_cmp.png")
    renderer.delete()

    print(f"\ndone {n} → {out}/  (no surface {no_surface}/{n}); PNG left=GT right=recon")


if __name__ == "__main__":
    main()
