"""Sample the VAE latent and decode → shapes. A latent-health / generation probe (RESEARCH_LOG:
"latent health for generation, sample-decode"). This is NOT the real generation path — unconditional
generation is the DiT's job; here we just look at what decoding a *sampled* latent produces.

Modes (+mode=):
  posterior (default): encode shape[shape_idx] → z ~ N(mu, temp*sigma) → decode. Reconstruction with
                       variation; +temp scales sigma to probe how smooth the latent is around mu.
  prior              : z ~ N(0, I) → decode. True unconditional gen — but the VAE latent is NOT trained
                       to match N(0,1) (small KL weight), so expect poor / broken samples. That gap is
                       exactly why a DiT is needed; this mode measures it.
  aggpost            : fit per-element mean/std of mu over +fit_shapes shapes, z ~ N(agg_mean, agg_std)
                       → decode. A cheap "aggregate posterior" prior proxy, usually far better than N(0,1).

Usage:
  uv run python scripts/sample_shapes.py +experiment=anchor_converge_lr3.5e-6_1024_l8_16 \
      +ckpt=/abs/best.pt +mode=posterior +shape_idx=0 +samples=8 [+temp=1.0] [+resolution=128]
  # meshes saved to results/samples/<mode>/<mode>_NN.obj
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

from src.engine.utils import get_device, set_seed
from src.model.shape.VAE.postprocess import Postprocess


def _encode(model, ds, i, device):
    it = ds[i]
    q = it["query"].unsqueeze(0).to(device)
    data = it["data"].unsqueeze(0).to(device)
    mu, logvar = model.encoder(q, data)
    return mu, torch.exp(0.5 * logvar)


def _build_renderer(w: int = 512, h: int = 512):
    """Headless pyrender (same camera/light as evaluate.py's viz), for +render=true."""
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import mesh_to_sdf  # noqa: F401 — must precede pyrender (it monkeypatches OpenGL)
    import numpy as np
    import pyrender

    def look_at(eye, target=(0, 0, 0), up=(0, 1, 0)):
        eye, target, up = map(lambda v: np.asarray(v, np.float32), (eye, target, up))
        f = target - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        p = np.eye(4, dtype=np.float32)
        p[:3, 0], p[:3, 1], p[:3, 2], p[:3, 3] = r, u, -f, eye
        return p

    renderer = pyrender.OffscreenRenderer(w, h)

    def render(mesh) -> "np.ndarray":
        if mesh is None:
            return np.zeros((h, w, 3), np.uint8)
        scene = pyrender.Scene(ambient_light=[0.35, 0.35, 0.35], bg_color=[0, 0, 0])
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        pose = look_at(eye=(2.0, 1.5, 2.6))  # 3/4 view of the [-1,1] cube
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0), pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        color, _ = renderer.render(scene)
        return color[..., :3]

    return renderer, render


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ckpt = cfg.ckpt if os.path.isabs(cfg.ckpt) else os.path.join(get_original_cwd(), cfg.ckpt)
    assert os.path.isfile(ckpt), f"ckpt not found: {ckpt}"
    set_seed(int(cfg.get("seed", 0)))
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"])

    ds = instantiate(cfg.data[cfg.get("split", "val")].dataset)
    post = Postprocess(ds.preprocessor.fourier_embedder, resolution=int(cfg.get("resolution", 128)))

    mode = cfg.get("mode", "posterior")
    n, temp = int(cfg.get("samples", 8)), float(cfg.get("temp", 1.0))
    out = os.path.join(get_original_cwd(), "results", "samples", mode)
    os.makedirs(out, exist_ok=True)
    renderer = _build_renderer() if bool(cfg.get("render", False)) else None

    with torch.no_grad():
        if mode == "aggpost":  # fit a Gaussian to mu across shapes → prior proxy
            fit = min(int(cfg.get("fit_shapes", 64)), len(ds))
            M = torch.cat([_encode(model, ds, i, device)[0] for i in range(fit)], 0)  # (fit, L, D)
            mu, std = M.mean(0, keepdim=True), M.std(0, keepdim=True)
            print(f"aggpost fit over {fit} shapes: mu-spread={std.mean():.4f}")
        else:  # posterior / prior both need one encode (for mu/sigma or just the latent shape)
            mu, std = _encode(model, ds, int(cfg.get("shape_idx", 0)), device)
            print(f"shape[{int(cfg.get('shape_idx', 0))}]: latent {tuple(mu.shape)}  sigma_rms={std.pow(2).mean().sqrt():.4f}")

        saved = 0
        for k in range(n):
            if mode == "prior":
                z = torch.randn_like(mu)
            else:  # posterior / aggpost: N(mu, temp*std)
                z = mu + temp * std * torch.randn_like(std)
            mesh = post.to_mesh(model.decode, z)  # decode(z, pe) → SDF grid → marching cubes
            if mesh is None:
                print(f"[{mode} {k:02d}] no zero-crossing surface (None)")
                continue
            p = os.path.join(out, f"{mode}_{k:02d}.obj")
            mesh.export(p)
            if renderer is not None:
                from PIL import Image
                Image.fromarray(renderer[1](mesh)).save(os.path.join(out, f"{mode}_{k:02d}.png"))
            saved += 1
            print(f"[{mode} {k:02d}] verts={len(mesh.vertices):>6} → {os.path.relpath(p, get_original_cwd())}")

    if renderer is not None:
        renderer[0].delete()
    print(f"\n{mode}: saved {saved}/{n} meshes to {os.path.relpath(out, get_original_cwd())}/  (temp={temp})")


if __name__ == "__main__":
    main()
