#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paper_eval import load_run, sample_fn
from src.model.shape.VAE.vae import Gaussian


def injected_noise(model, mu, logvar):
    """What the decoder actually receives as noise scale.

    Plain VAE: sigma. lambda-VAE: exp(min(log sigma, lam*log sigma)), i.e. the same elementwise
    min taken in log space that Gaussian.sample uses, so this matches training rather than
    reporting sigma and calling it the injected noise.
    """
    g = Gaussian(mu, logvar)
    if float(getattr(model, "lam_delta", 0.0)) <= 1.0:
        return g.std
    lam = model._lam(g.std)
    log_std = torch.log(g.std.clamp_min(1e-12))
    return torch.exp(torch.minimum(log_std, lam * log_std))


class Accum:
    """Per-channel sums, in float64.

    Two notions of "signal" are kept because they answer different questions and the figure's
    original script is gone, so neither can be assumed:
      pooled  - std of mu over (shapes x tokens), the literal mu.std() per channel
      byshape - Var over shapes per (token, channel), then averaged over tokens; this is the
                Active Units convention and counts only shape-to-shape variation
    """

    def __init__(self, tokens, channels, device):
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        self.n = 0
        self.mu_s, self.mu_s2 = z(channels), z(channels)          # pooled over tokens+shapes
        self.el_s, self.el_s2 = z(tokens, channels), z(tokens, channels)
        self.noise_s = z(channels)
        self.tokens = tokens

    def add(self, mu, noise):
        m, nz = mu.double(), noise.double()
        self.mu_s += m.sum(0)
        self.mu_s2 += (m * m).sum(0)
        self.el_s += m
        self.el_s2 += m * m
        self.noise_s += nz.mean(0)
        self.n += 1

    def result(self):
        n, T = self.n, self.tokens
        pooled_mean = self.mu_s / (n * T)
        pooled_var = self.mu_s2 / (n * T) - pooled_mean ** 2
        el_mean = self.el_s / n
        el_var = (self.el_s2 - n * el_mean ** 2) / (n - 1)        # unbiased, over shapes
        return {
            "signal_pooled": pooled_var.clamp_min(0).sqrt().cpu(),
            "signal_byshape": el_var.mean(0).clamp_min(0).sqrt().cpu(),
            "noise": (self.noise_s / n).cpu(),
        }


def ratio(v):
    v = v[v > 0]
    return float(v.max() / v.min()) if v.numel() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--shapes", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", default="results/channel_usage.json")
    ap.add_argument("--fig-out", default="docs/figures/per_channel_usage.png")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    runs = [c.split("=", 1) for c in args.ckpt]
    for _, p in runs:
        if not os.path.isfile(p):
            raise SystemExit(f"checkpoint not found: {p}")

    mesh_paths = sorted(glob.glob(os.path.join("data", args.split, "**", "*.off.gz"), recursive=True))
    if not mesh_paths:
        raise SystemExit(f"no meshes under data/{args.split}")
    n = len(mesh_paths) if args.shapes <= 0 else min(args.shapes, len(mesh_paths))
    pick = sorted(np.random.default_rng(args.seed).choice(len(mesh_paths), n, replace=False))
    rels = [os.path.splitext(os.path.relpath(mesh_paths[i], os.path.join("data", args.split))
                             .removesuffix(".gz"))[0] for i in pick]
    work = [(r, os.path.join("cache", r.split(os.sep)[0], "4_watertight_scaled",
                             os.path.basename(r) + ".pt")) for r in rels]
    work = [(r, c) for r, c in work if os.path.isfile(c)]
    if not work:
        raise SystemExit("none of the selected shapes have a cache file")
    print(f"{len(work)}/{n} shapes from data/{args.split} have a cache; using those")

    out = {}
    for name, ckpt in runs:
        t0 = time.time()
        model, pre, cfg, epoch = load_run(ckpt, args.device)
        smp = sample_fn(pre, cfg)
        acc = None
        for rel, cpath in tqdm(work, desc=f"{name} (epoch {epoch})", unit="shape"):
            try:
                cache = torch.load(cpath, weights_only=False)
                torch.manual_seed(args.seed)          # identical query points for every checkpoint
                q, d, *_ = smp(cache)
                with torch.no_grad():
                    mu, logvar = model.encoder(q[None].to(args.device), d[None].to(args.device))
                    nz = injected_noise(model, mu, logvar)
                if acc is None:
                    acc = Accum(mu.shape[1], mu.shape[2], args.device)
                acc.add(mu[0], nz[0])
            except Exception as e:
                print(f"\n  skip {rel}: {type(e).__name__}: {e}", flush=True)
        if acc is None or acc.n < 2:
            raise SystemExit(f"{name}: not enough shapes encoded")
        r = acc.result()
        snr = r["signal_pooled"] / r["noise"].clamp_min(1e-12)
        snr_bs = r["signal_byshape"] / r["noise"].clamp_min(1e-12)
        out[name] = {
            "n": acc.n, "epoch": epoch, "ckpt": ckpt, "seconds": round(time.time() - t0, 1),
            "signal_pooled": r["signal_pooled"].tolist(),
            "signal_byshape": r["signal_byshape"].tolist(),
            "noise": r["noise"].tolist(),
            "signal_pooled_ratio": ratio(r["signal_pooled"]),
            "signal_byshape_ratio": ratio(r["signal_byshape"]),
            "noise_ratio": ratio(r["noise"]),
            "noise_mean": float(r["noise"].mean()),
            "snr_median": float(snr.median()),
            "snr_byshape_median": float(snr_bs.median()),
        }
        o = out[name]
        print(f"  {name}: signal max/min {o['signal_pooled_ratio']:.1f}x  "
              f"noise mean {o['noise_mean']:.3f} (max/min {o['noise_ratio']:.1f}x)  "
              f"SNR median {o['snr_median']:.2f}   [{o['seconds']}s]")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json_out}")

    names = list(out)
    ns = out[names[0]]["n"]
    ch = len(out[names[0]]["noise"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle(f"Per-channel latent usage ({ns} shapes, latent_dim={ch})", fontsize=12)
    colours = {"vanilla": "#1f77b4", "disjoint": "#ff7f0e", "lambda": "#2ca02c"}
    for k in names:
        o, c = out[k], colours.get(k)
        sig = np.sort(np.array(o["signal_pooled"]))[::-1]
        noi = np.sort(np.array(o["noise"]))
        snr = np.sort(np.array(o["signal_pooled"]) / np.maximum(o["noise"], 1e-12))[::-1]
        ax[0].plot(sig, label=f"{k}  (max/min {o['signal_pooled_ratio']:.1f}x)", color=c)
        ax[1].plot(noi, label=f"{k}  (mean {o['noise_mean']:.3f})", color=c)
        ax[2].plot(snr, label=f"{k}  (median {o['snr_median']:.2f})", color=c)
    ax[0].set_title("(a) signal: std of mu per channel")
    ax[0].set_ylabel("std of mu")
    ax[1].set_title("(b) injected noise per channel\n(sigma, or sigma^lambda for lambda-VAE)")
    ax[1].set_ylabel("noise std")
    ax[2].set_title("(c) per-channel SNR = mu.std / noise")
    ax[2].set_ylabel("SNR")
    ax[2].axhline(1.0, ls="--", lw=0.9, color="#444444")
    ax[2].annotate("SNR = 1", (0.02, 1.0), xycoords=("axes fraction", "data"),
                   va="bottom", fontsize=8, color="#444444")
    for a in ax:
        a.set_xlabel("channel (sorted)")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.fig_out) or ".", exist_ok=True)
    fig.savefig(args.fig_out, dpi=150)
    print(f"wrote {args.fig_out}")


if __name__ == "__main__":
    main()

"""
Per-channel signal, injected noise and SNR of the VAE latent, over a whole split.

    uv run python -m scripts.channel_usage \
        --ckpt vanilla=/abs/best.pt --ckpt disjoint=/abs/best.pt --ckpt lambda=/abs/best.pt \
        --shapes 0 --device cuda

Needs only the encoder, so no decoder and no marching cubes. Writes results/channel_usage.json
and redraws docs/figures/per_channel_usage.png.
"""
