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
from src.metrics import SIOU, VIOU, ChamferDistance, FScore, NormalConsistency

_SURF_SAMPLES = 50_000  # ponytail: surface points per cloud for chamfer/f-score/NC; stable yet cheap


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

    def sample_pn(mesh, n: int):
        """Surface points + per-point normals (from the sampled face), for chamfer/f-score/NC."""
        pts, fi = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts), np.asarray(mesh.face_normals[fi])

    out = os.path.join(get_original_cwd(), "results", cfg.get("name", "recon"))  # project root, not the hydra run dir
    post = Postprocess(ds.preprocessor.fourier_embedder, resolution=int(cfg.get("resolution", 128)))
    return SimpleNamespace(post=post, renderer=renderer, out=out, load_gt=load_gt,
                           save_pair=save_pair, sample_pn=sample_pn)


def _preflight(cfg: DictConfig) -> None:
    """Validate the error-prone args up front so a bad path/number fails in seconds,
    not after minutes of eval. Also resolves a relative +ckpt against the original cwd
    (hydra chdirs into the run dir, which silently breaks bare relative paths)."""
    from hydra.utils import get_original_cwd

    if "ckpt" not in cfg:
        raise SystemExit("missing +ckpt=<path to a .pt checkpoint>")
    ckpt = cfg.ckpt if os.path.isabs(cfg.ckpt) else os.path.join(get_original_cwd(), cfg.ckpt)
    if not os.path.isfile(ckpt):
        raise SystemExit(f"ckpt not found: {ckpt}")
    cfg.ckpt = ckpt  # write back the resolved absolute path so torch.load below just works

    split = cfg.get("split", "val")  # +split=test to eval on the paper test set; default val
    obj_root = cfg.data[split].dataset.obj_root
    if not os.path.isdir(obj_root):
        raise SystemExit(f"{split} obj_root not found: {obj_root}")

    shapes, samples = int(cfg.get("shapes", 16)), int(cfg.get("samples", 0))
    resolution, fscore_tau = int(cfg.get("resolution", 128)), float(cfg.get("fscore_tau", 0.02))
    if shapes <= 0 or samples < 0 or resolution <= 0 or fscore_tau <= 0:
        raise SystemExit(f"bad numeric arg: shapes={shapes} samples={samples} "
                         f"resolution={resolution} fscore_tau={fscore_tau}")
    if bool(cfg.get("viz", False)) and fscore_tau <= 2 / resolution:
        raise SystemExit(f"fscore_tau={fscore_tau} must exceed MC cell size 2/{resolution}"
                         f"={2 / resolution:.4f}, else it scores below discretisation noise")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _preflight(cfg)  # fail fast on bad ckpt/paths/args before any heavy work
    set_seed(0)
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    model.load_state_dict(torch.load(cfg.ckpt, map_location=device, weights_only=False)["model"])

    ds = instantiate(cfg.data[cfg.get("split", "val")].dataset)
    n = min(int(cfg.get("shapes", 16)), len(ds))
    viz = bool(cfg.get("viz", False))
    v = _viz_setup(cfg, ds) if viz else None

    # +samples=K averages K decoded fields from z~N(mu,sigma) instead of decode(mu). Discriminates
    # posterior collapse (sample-avg still bad → info gone) from mu being an off-manifold point
    # (sample-avg recovers → info present, mu is just a bad point in the high-dim latent).
    K = int(cfg.get("samples", 0))

    # tau must exceed the marching-cubes cell size (2/res); at res 128 that's ~0.0156, so 0.01 would
    # score agreement finer than a vertex can be placed (discretisation-noise floor). Default above it.
    fscore_tau = float(cfg.get("fscore_tau", 0.02))
    preds, gts = [], []
    chamfers, fscores, ncs = [], [], []  # per-shape mesh metrics (only shapes with an extracted surface)
    no_surface = 0
    mu_vecs, sig_rms = [], []  # per-shape flattened mu and rms(sigma): mu-spread=collapse, sigma=shell size
    # Volume V-IoU (paper protocol): occupancy IoU over uniform-in-volume points. sample() shuffles+subsets
    # the SDF bank, destroying the uniform-point identity, so we read the cache's unshuffled uniform tail
    # (last n_query_uniform pts, see preprocess.py) and decode it separately in the loop below.
    n_uni = int(getattr(ds.preprocessor, "n_query_uniform", 0) or 0)
    vol_pts = int(cfg.get("vol_pts", 16384))   # uniform pts decoded per shape for the volume-IoU
    vol_preds, vol_gts = [], []
    with torch.no_grad():
        for i in range(n):
            it = ds[i]
            q = it["query"].unsqueeze(0).to(device)
            data = it["data"].unsqueeze(0).to(device)
            qp = it["query_points"].unsqueeze(0).to(device)
            mu, logvar = model.encoder(q, data)
            std = torch.exp(0.5 * logvar)
            mu_vecs.append(mu.flatten().cpu())
            sig_rms.append(std.pow(2).mean().sqrt().item())
            if K > 0:
                fields = [model.decode(mu + std * torch.randn_like(std), qp)[0].squeeze(0) for _ in range(K)]
                pred = torch.stack(fields).mean(0).cpu()
            else:
                pred = model.decode(mu, qp)[0].squeeze(0).cpu()
            preds.append(pred)
            gts.append(it["gt_sdf"])

            if n_uni:  # volume-IoU: decode the cache's unshuffled uniform tail (paper's in-3D-space points)
                c = torch.load(ds._cache_path(ds.paths[i]), weights_only=False)
                uxyz, ug = c["sdf_query_points"][-n_uni:], c["gt_sdf"][-n_uni:].flatten()
                if uxyz.shape[0] > vol_pts:
                    sel = torch.randperm(uxyz.shape[0])[:vol_pts]
                    uxyz, ug = uxyz[sel], ug[sel]
                upe = ds.preprocessor.fourier_embedder(uxyz).unsqueeze(0).to(device)
                if K > 0:
                    uf = torch.stack([model.decode(mu + std * torch.randn_like(std), upe)[0].squeeze(0) for _ in range(K)]).mean(0)
                else:
                    uf = model.decode(mu, upe)[0].squeeze(0)
                vol_preds.append(uf.flatten().cpu()); vol_gts.append(ug)

            if viz:
                if K > 0:  # sample-avg MC: decode(mu) is off-manifold when the posterior is mu-broken, so
                    # average K fixed latent draws per grid point (same K-sample logic as the SDF metric above)
                    zs = [mu + std * torch.randn_like(std) for _ in range(K)]
                    avg_decode = lambda _z, pe: (torch.stack([model.decode(zk, pe)[0] for zk in zs]).mean(0), None)
                    recon = v.post.to_mesh(avg_decode, mu)
                else:
                    recon = v.post.to_mesh(model.decode, mu)
                gt_mesh = v.load_gt(ds.paths[i])
                rel = os.path.splitext(os.path.relpath(ds.paths[i], ds.obj_root).removesuffix(".gz"))[0]
                v.save_pair(os.path.join(v.out, rel), gt_mesh, recon)
                if recon is None:
                    no_surface += 1  # chamfer/f-score/NC uncomputable for this shape → recorded as N/A below
                    print(f"[{i}] {rel}  no zero-crossing surface (None) → chamfer/f-score/NC = N/A")
                else:
                    rp, rn = v.sample_pn(recon, _SURF_SAMPLES)
                    gp, gn = v.sample_pn(gt_mesh, _SURF_SAMPLES)
                    cd = ChamferDistance()(rp, gp)
                    fs = FScore(tau=fscore_tau)(rp, gp)
                    nc = NormalConsistency()(rp, rn, gp, gn)
                    chamfers.append(cd); fscores.append(fs); ncs.append(nc)
                    print(f"[{i}] {rel}  verts={len(recon.vertices)}  chamfer={cd:.5f}  "
                          f"f-score={fs:.4f}  NC={nc:.4f}")

    import math
    import statistics as _st

    pred = torch.cat(preds).flatten()
    gt = torch.cat(gts).flatten()
    v_iou = VIOU()(pred, gt)          # occupancy IoU over all query points (our bank is ~80% near-surface)
    s_iou = SIOU()(pred, gt)          # occupancy IoU restricted to the near-surface band

    # Volume V-IoU (paper protocol): occupancy IoU over the uniform-in-volume points decoded per shape above.
    v_iou_uniform = VIOU()(torch.cat(vol_preds), torch.cat(vol_gts)) if vol_preds else float("nan")

    def _iou_str(x: float) -> str:
        return "N/A (empty band)" if math.isnan(x) else f"{x * 100:.2f}%"

    def _mesh_agg(vals: list[float]) -> str:  # mean over shapes with a surface; explicit N/A when none
        good = [x for x in vals if not math.isnan(x)]
        return f"N/A (0/{n} shapes)" if not good else f"{sum(good) / len(good):.5f}  ({len(good)}/{n} shapes)"

    mode = f"sample-avg K={K}" if K > 0 else "mu-path"
    print(f"\nshapes={n}  points={gt.numel():,}  eval={mode}  ckpt={cfg.ckpt}")
    mu_spread = torch.stack(mu_vecs).std(0).mean().item()  # per-elem std across shapes, averaged
    d = mu_vecs[0].numel()
    print(f"sigma_rms        : {_st.mean(sig_rms):.4f}  shell radius ~ {_st.mean(sig_rms) * d**0.5:.1f} "
          f"(sigma_rms*sqrt({d}))  [posterior diagnostic]")
    print(f"mu-spread        : {mu_spread:.5f}  per-elem std across shapes  (near 0 = collapse)  [diagnostic]")
    print(f"overall RMS      : {(pred - gt).pow(2).mean().sqrt():.4f}  (SDF scale ~[-1,1])  [diagnostic]")
    print(f"V-IoU (all pts)  : {_iou_str(v_iou)}  [our default: ~80% near-surface]")
    print(f"V-IoU (uniform)  : {_iou_str(v_iou_uniform)}  [paper volume-IoU: {min(n_uni, vol_pts)} uniform-in-vol pts/shape]")
    print(f"S-IoU (|gt|<0.02): {_iou_str(s_iou)}")
    if viz:
        print(f"chamfer          : {_mesh_agg(chamfers)}   (no surface {no_surface}/{n} → N/A)")
        print(f"f-score@{fscore_tau:g}     : {_mesh_agg(fscores)}")
        print(f"normal-consist.  : {_mesh_agg(ncs)}")
        v.renderer.delete()
        print(f"\nviz → {v.out}/  (PNG left=GT right=recon)")
    else:
        print("chamfer/f-score/NC: add +viz=true to extract meshes and compute these")


if __name__ == "__main__":
    main()
