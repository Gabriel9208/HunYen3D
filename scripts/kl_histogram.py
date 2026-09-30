import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.model.shape.VAE.vae import Gaussian
from src.model.shape.VAE.encoder import DoubleStreamEncoder, FrameEncoder

DEAD_KL = 0.01  # standard "active unit" threshold from the beta-VAE literature


def kl_breakdown(mu: torch.Tensor, logvar: torch.Tensor) -> dict:
    g = Gaussian(mu, logvar)
    kl = 0.5 * (g.mu.pow(2) + g.var - 1.0 - g.logvar)   # (B, N, D)
    return {
        "per_channel": kl.mean(dim=(0, 1)),   # (D,)
        "per_token": kl.mean(dim=(0, 2)),     # (N,)
        "flat": kl.flatten(),
    }


def plot(bd: dict, out_path: str) -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4))

    pc = bd["per_channel"].numpy()
    a.bar(range(len(pc)), pc)
    a.axhline(DEAD_KL, color="red", linestyle="--", linewidth=0.8)
    a.set_title("per-channel KL")
    a.set_xlabel("channel index")
    a.set_ylabel("mean KL")

    pt = bd["per_token"].numpy()
    b.bar(range(len(pt)), pt)
    b.axhline(DEAD_KL, color="red", linestyle="--", linewidth=0.8)
    b.set_title("per-token KL")
    b.set_xlabel("token index")
    b.set_ylabel("mean KL")

    c.hist(bd["flat"].numpy(), bins=100)
    c.set_yscale("log")
    c.set_title("global KL histogram")
    c.set_xlabel("KL (per element)")
    c.set_ylabel("count (log)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--experiment", help="name under configs/experiment/, composed via +experiment=")
    g.add_argument("--config", help="a run's own outputs/.../.hydra/config.yaml, fully resolved -- "
                    "use this when the experiment's defaults chain is broken (a base config it "
                    "inherits from is missing), since the run's own resolved copy has no references left")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shapes", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0, help="seeds the FPS/downsample randomness inside "
                     "pre.frame_sample()/double_stream_sample()/sample() -- those don't take a generator "
                     "argument, so this seeds the global RNG once before the sampling loop instead")
    ap.add_argument("--tag", default="", help="extra token appended to the output filename -- two runs "
                     "of the same experiment share cfg.wandb.name, so without this the second run's "
                     "epoch-N image silently overwrites the first run's")
    ap.add_argument("--permute-query", action="store_true", help="apply one fixed random permutation "
                     "(seeded by --seed) to the query tokens (q_xyz/q_emb) of every shape, before "
                     "encoding -- tests whether a per-token KL anomaly follows the point's content "
                     "(moves to the new index) or is tied to raw sequence position (stays put)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if args.config:
        cfg = OmegaConf.load(args.config)
    else:
        with initialize(version_base=None, config_path="../configs"):
            cfg = compose("config", overrides=[f"+experiment={args.experiment}"])
    model = instantiate(cfg.model)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd["model"])
    model.eval().to(args.device)
    pre = instantiate(cfg.preprocess)

    is_double_stream = isinstance(model.encoder, DoubleStreamEncoder)
    is_frame = isinstance(model.encoder, FrameEncoder)

    query_perm = torch.randperm(cfg.model.num_latents) if args.permute_query else None
    if query_perm is not None:
        print(f"--permute-query: applying one fixed permutation (seed={args.seed}) to every shape's query")

    all_paths = sorted(glob.glob("cache/*/4_watertight_scaled/*.pt"))
    stride = max(1, len(all_paths) // args.shapes)
    mus, logvars = [], []
    for p in all_paths[::stride][: args.shapes]:
        cache = torch.load(p, weights_only=False)
        with torch.no_grad():
            if is_frame:
                q_xyz, d_xyz, _, _, q_emb, d_emb, *_ = pre.frame_sample(cache)
                if query_perm is not None:
                    q_xyz, q_emb = q_xyz[query_perm], q_emb[query_perm]
                mu, logvar = model.encoder(
                    q_xyz.unsqueeze(0).to(args.device), d_xyz.unsqueeze(0).to(args.device),
                    q_emb.unsqueeze(0).to(args.device), d_emb.unsqueeze(0).to(args.device),
                )
            elif is_double_stream:
                q_xyz, d_xyz, _, _, q_emb, d_emb, *_ = pre.double_stream_sample(cache)
                if query_perm is not None:
                    q_xyz, q_emb = q_xyz[query_perm], q_emb[query_perm]
                mu, logvar = model.encoder(
                    q_xyz.unsqueeze(0).to(args.device), d_xyz.unsqueeze(0).to(args.device),
                    q_emb.unsqueeze(0).to(args.device), d_emb.unsqueeze(0).to(args.device),
                )
            else:
                q, d, *_ = pre.sample(cache)
                if query_perm is not None:
                    q = q[query_perm]
                mu, logvar = model.encoder(q.unsqueeze(0).to(args.device), d.unsqueeze(0).to(args.device))
        mus.append(mu[0].cpu())
        logvars.append(logvar[0].cpu())
    mu = torch.stack(mus)          # (B, N, D)
    logvar = torch.stack(logvars)  # (B, N, D)

    bd = kl_breakdown(mu, logvar)
    pc, pt = bd["per_channel"], bd["per_token"]
    n_dead_ch, n_dead_tok = int((pc < DEAD_KL).sum()), int((pt < DEAD_KL).sum())

    epoch = sd.get("epoch", "?")
    name = args.experiment or cfg.wandb.name
    suffix = (f"_epoch{epoch}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
              + ("_permuted" if args.permute_query else ""))
    out_path = os.path.join("results", "kl_histograms", f"{name}{suffix}.png")
    plot(bd, out_path)

    print(f"\nepoch={epoch}  mu/logvar shape = {tuple(mu.shape)}  (B, N, D)")
    print(f"per-channel KL  min/mean/max = {pc.min():.4f}/{pc.mean():.4f}/{pc.max():.4f}  "
          f"dead(<{DEAD_KL}) = {n_dead_ch}/{pc.numel()}")
    print(f"per-token   KL  min/mean/max = {pt.min():.4f}/{pt.mean():.4f}/{pt.max():.4f}  "
          f"dead(<{DEAD_KL}) = {n_dead_tok}/{pt.numel()}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

"""
    uv run python -m scripts.kl_histogram --experiment vae_rope_kl1e-4_1024_l8_16 --ckpt <path>
"""