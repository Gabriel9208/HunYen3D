from __future__ import annotations

import argparse
import glob
import gzip
import os
import sys

import torch

HUNYUAN_ROOT = "/home/gabriel/dev/PhysicsInTheLoop/third_party/Hunyuan3D-2.1/hy3dshape"


def _load_mesh(path):
    import trimesh
    if path.endswith(".gz"):
        inner = os.path.splitext(path[:-3])[1].lstrip(".")
        with gzip.open(path, "rb") as f:
            return trimesh.load(f, file_type=inner, force="mesh")
    return trimesh.load_mesh(path, force="mesh")


def _stats(mu: torch.Tensor, sigma: torch.Tensor) -> dict:
    mu, sigma = mu.float(), sigma.float()
    mv, sv = mu.var().item(), (sigma ** 2).mean().item()
    return {
        "mu_std": mu.std().item(),
        "sigma": sigma.mean().item(),
        "snr": mu.std().item() / sigma.mean().item(),
        "signal_frac": mv / (mv + sv),
    }


def _report(tag: str, s: dict, disagree: float | None) -> None:
    line = (f"{tag:<24} mu.std={s['mu_std']:.4f}  sigma={s['sigma']:.4f}  "
            f"SNR={s['snr']:.3f}  signal={s['signal_frac']*100:5.1f}%")
    if disagree is not None:
        line += f"  mu-vs-sample field disagreement={disagree*100:.1f}%"
    print(line)


def run_ours(paths, experiment, ckpt):
    from hydra import compose, initialize
    from hydra.utils import instantiate

    with initialize(version_base=None, config_path="../configs"):
        cfg = compose("config", overrides=[f"+experiment={experiment}"])
    vae = instantiate(cfg.model)
    vae.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    vae.eval()
    pre = instantiate(cfg.preprocess)

    mus, sigs, dis = [], [], []
    with torch.no_grad():
        for p in paths:
            cache = torch.load(p, weights_only=False)
            q, d, qp, _, _ = pre.sample(cache)
            mu, logvar = vae.encoder(q.unsqueeze(0), d.unsqueeze(0))
            sigma = torch.exp(0.5 * torch.clamp(logvar, -30, 20))
            mus.append(mu.flatten()); sigs.append(sigma.flatten())

            torch.manual_seed(0)
            f_mu, _ = vae.decode(mu, qp.unsqueeze(0))
            f_s, _ = vae.decode(mu + torch.randn_like(sigma) * sigma, qp.unsqueeze(0))
            dis.append(((f_mu - f_s).std() / f_mu.std()).item())
    return _stats(torch.cat(mus), torch.cat(sigs)), sum(dis) / len(dis)


def _register_bare_package():
    # avoid import .pipelines/.postprocessors/.preprocessors in hy3dshape/__init__.py 
    import types
    if "hy3dshape" in sys.modules:
        return
    pkg = types.ModuleType("hy3dshape")
    pkg.__path__ = [os.path.join(HUNYUAN_ROOT, "hy3dshape")]
    sys.modules["hy3dshape"] = pkg


def _fps_fallback(src, batch=None, ratio=None, random_start=True, batch_size=None, ptr=None):
    # drop-in for torch_cluster.fps
    src = src.float()
    if batch is None:
        batch = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
    out = []
    for b in batch.unique():
        idx = (batch == b).nonzero(as_tuple=True)[0]
        pts = src[idx]
        n = pts.shape[0]
        k = max(1, int(round((ratio if ratio is not None else 1.0) * n)))
        sel = torch.empty(k, dtype=torch.long, device=src.device)
        min_d = torch.full((n,), float("inf"), device=src.device)
        cur = int(torch.randint(n, ())) if random_start else 0
        for i in range(k):
            sel[i] = cur
            min_d = torch.minimum(min_d, (pts - pts[cur]).pow(2).sum(-1))
            cur = int(torch.argmax(min_d))
        out.append(idx[sel])
    return torch.cat(out)


def run_hunyuan(paths, n_uniform, n_sharp, device):
    if HUNYUAN_ROOT not in sys.path:
        sys.path.insert(0, HUNYUAN_ROOT)
    _register_bare_package()
    from hy3dshape.models.autoencoders import attention_blocks
    attention_blocks.fps = _fps_fallback
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.models.autoencoders.model import DiagonalGaussianDistribution
    from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader

    dtype = torch.float16 if device == "cuda" else torch.float32
    vae = ShapeVAE.from_pretrained("tencent/Hunyuan3D-2.1", use_safetensors=False,
                                    variant="fp16", device=device, dtype=dtype)
    vae.eval()
    # Their own demo's setting; also what their shipped config ships (pc_sharpedge_size: 0).
    loader = SharpEdgeSurfaceLoader(num_uniform_points=n_uniform, num_sharp_points=n_sharp)

    mus, sigs, dis = [], [], []
    with torch.no_grad():
        for p in paths:
            surface = loader(_load_mesh(p)).to(device, dtype=dtype)
            pc, feats = surface[:, :, :3], surface[:, :, 3:]
            latents, _ = vae.encoder(pc, feats)
            posterior = DiagonalGaussianDistribution(vae.pre_kl(latents), feat_dim=-1)
            mus.append(posterior.mean.flatten().float().cpu())
            sigs.append(posterior.std.flatten().float().cpu())

            # GT-free: same query points, decode from mu vs from a posterior sample.
            qp = (torch.rand(1, 20000, 3, device=device, dtype=dtype) * 2 - 1)
            f_mu = vae.geo_decoder(queries=qp, latents=vae.decode(posterior.mode())).float()
            f_s = vae.geo_decoder(queries=qp, latents=vae.decode(posterior.sample())).float()
            dis.append(((f_mu - f_s).std() / f_mu.std()).item())
    return _stats(torch.cat(mus), torch.cat(sigs)), sum(dis) / len(dis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", type=int, default=5)
    ap.add_argument("--experiment", default="vae_converge_kl1e-4_lr3.5e-6_1024_l8_16")
    ap.add_argument("--ckpt", default="/data/outputs/vae_converge_kl1e-4_lr3.5e-6_1024_l8_16/2026-08-12_16-49-28/checkpoints/best.pt")
    ap.add_argument("--n-uniform", type=int, default=81920, help="Hunyuan loader uniform points (their demo uses 81920)")
    ap.add_argument("--n-sharp", type=int, default=0, help="Hunyuan loader sharp points (their demo + shipped config use 0)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ours-only", action="store_true", help="skip Hunyuan (no checkpoint download)")
    ap.add_argument("--hunyuan-only", action="store_true")
    args = ap.parse_args()

    cache_paths = sorted(glob.glob("cache/*/4_watertight_scaled/*.pt"))[: args.shapes]
    mesh_paths = sorted(glob.glob("data/*/*/4_watertight_scaled/*.off.gz"))[: args.shapes]
    if not args.hunyuan_only and not cache_paths:
        raise SystemExit("no cache/*.pt found for our VAE")
    if not args.ours_only and not mesh_paths:
        raise SystemExit("no data/**/*.off.gz meshes found for the Hunyuan loader")

    print(f"signal = fraction of latent variance carrying shape info; the rest is posterior noise\n")
    if not args.hunyuan_only:
        s, d = run_ours(cache_paths, args.experiment, args.ckpt)
        _report("ours (" + args.experiment.split("_")[1] + ")", s, d)
    if not args.ours_only:
        s, d = run_hunyuan(mesh_paths, args.n_uniform, args.n_sharp, args.device)
        _report("hunyuan3d-2.1 (released)", s, d)


if __name__ == "__main__":
    main()

"""
    uv run python -m scripts.hunyuan_posterior_check --shapes 5
    uv run python -m scripts.hunyuan_posterior_check --shapes 5 --ours-only   # skip the download
"""