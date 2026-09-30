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

from scripts.paper_eval import load_run, sample_fn

AU_THRESHOLD = 0.01


class VarAccumulator:
    def __init__(self, shape, device):
        self.n = 0
        self.s = torch.zeros(shape, dtype=torch.float64, device=device)
        self.s2 = torch.zeros(shape, dtype=torch.float64, device=device)

    def add(self, mu: torch.Tensor) -> None:
        m = mu.to(torch.float64)
        self.s += m
        self.s2 += m * m
        self.n += 1

    def variance(self) -> torch.Tensor:
        if self.n < 2:
            raise RuntimeError(f"need at least 2 shapes, got {self.n}")
        mean = self.s / self.n
        # ddof=1: an unbiased estimate, so the number does not drift with the shape count.
        return (self.s2 - self.n * mean * mean) / (self.n - 1)


def summarise(var: torch.Tensor, n_shapes: int) -> dict:
    """var: (tokens, channels) per-element Var_x(mu)."""
    v = var.detach().cpu()
    flat = v.flatten()
    per_token = v.mean(dim=1)      # average over channels
    per_channel = v.mean(dim=0)    # average over tokens
    pr = (flat.sum() ** 2 / (flat * flat).sum()).item()
    q = torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95], dtype=flat.dtype)
    return {
        "n": n_shapes,
        "elements": flat.numel(),
        "tokens": per_token.numel(),
        "channels": per_channel.numel(),
        "pass_elem": int((flat > AU_THRESHOLD).sum()),
        "pass_elem_frac": float((flat > AU_THRESHOLD).float().mean()),
        "pass_token": int((per_token > AU_THRESHOLD).sum()),
        "pass_channel": int((per_channel > AU_THRESHOLD).sum()),
        "elem_median": float(flat.median()),
        "elem_min": float(flat.min()),
        "elem_max": float(flat.max()),
        "token_median": float(per_token.median()),
        "channel_median": float(per_channel.median()),
        "participation_ratio": pr,
        "pr_frac": pr / flat.numel(),
        "total_var": float(flat.sum()),
        "elem_pct": {f"p{int(x*100)}": float(torch.quantile(flat, x)) for x in q},
        # the split that matters here: tokens 0-511 are the uniform anchors and 512-1023 the sharp
        # ones, in BOTH samplers (preprocess.py concatenates [surf_fps, sharp_fps] either way).
        "token_uni_mean": float(per_token[: per_token.numel() // 2].mean()),
        "token_sharp_mean": float(per_token[per_token.numel() // 2 :].mean()),
        "pass_token_uni": int((per_token[: per_token.numel() // 2] > AU_THRESHOLD).sum()),
        "pass_token_sharp": int((per_token[per_token.numel() // 2 :] > AU_THRESHOLD).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable; NAME labels the row")
    ap.add_argument("--shapes", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", default="results/active_units.json")
    ap.add_argument("--trace-every", type=int, default=0,
                    help="if >0, also record the running result every N shapes, so the reader can "
                         "see whether the count has converged in the shape dimension")
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
        acc, trace = None, []
        bar = tqdm(work, desc=f"{name} (epoch {epoch})", unit="shape")
        for i, (rel, cpath) in enumerate(bar, 1):
            try:
                cache = torch.load(cpath, weights_only=False)
                torch.manual_seed(args.seed)
                q, d, *_ = smp(cache)
                with torch.no_grad():
                    mu, _ = model.encoder(q[None].to(args.device), d[None].to(args.device))
                mu = mu[0]                                  # (tokens, channels)
                if acc is None:
                    acc = VarAccumulator(mu.shape, args.device)
                acc.add(mu)
            except Exception as e:                          # one bad cache must not lose the run
                print(f"\n  skip {rel}: {type(e).__name__}: {e}", flush=True)
                continue
            if args.trace_every and i % args.trace_every == 0 and acc.n >= 2:
                trace.append(summarise(acc.variance(), acc.n))
                bar.set_postfix(elem=trace[-1]["pass_elem"], tok=trace[-1]["pass_token"])
        if acc is None or acc.n < 2:
            raise SystemExit(f"{name}: not enough shapes encoded")
        rec = summarise(acc.variance(), acc.n)
        rec["epoch"], rec["ckpt"], rec["seconds"] = epoch, ckpt, round(time.time() - t0, 1)
        if trace:
            rec["trace"] = trace
        out[name] = rec
        print(f"  {name}: {rec['pass_elem']:,}/{rec['elements']:,} elements "
              f"({rec['pass_elem_frac']*100:.1f}%), {rec['pass_token']}/{rec['tokens']} tokens, "
              f"{rec['pass_channel']}/{rec['channels']} channels, "
              f"PR {rec['pr_frac']*100:.1f}%  [{rec['seconds']}s]")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json_out}")

    w = max(len(k) for k in out)
    print(f"\n{'model':<{w}} {'elements':>16} {'per-token':>12} {'per-channel':>12} {'PR':>8}")
    for k, r in out.items():
        print(f"{k:<{w}} {r['pass_elem']:>8,}/{r['elements']:,} "
              f"({r['pass_elem_frac']*100:4.1f}%) {r['pass_token']:>6}/{r['tokens']} "
              f"{r['pass_channel']:>7}/{r['channels']} {r['pr_frac']*100:7.1f}%")


if __name__ == "__main__":
    main()

"""
    uv run python -m scripts.active_units \
        --ckpt vanilla=/abs/best.pt --ckpt disjoint=/abs/best.pt --ckpt lambda=/abs/best.pt \
        --shapes 0 --device cuda
"""