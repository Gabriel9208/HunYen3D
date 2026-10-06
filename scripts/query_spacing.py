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

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

BLOCKS = ("all", "uni", "sharp", "cross")
STATS = ("min", "mean", "max")


def spacing(q: torch.Tensor, n_uni: int) -> dict[str, float]:
    """Nearest-neighbour spacing of one shape's query points.

    q is (num_latents, 3) with the first n_uni rows uniform and the rest sharp -- the order both
    samplers concatenate in (preprocess.py: cat([surf_fps, sharp_fps])).

    Within a block the nearest neighbour excludes the point itself; "cross" is each uniform point's
    distance to the nearest sharp point, so it needs no self-exclusion. Each block yields the
    min/mean/max over its per-point nearest-neighbour distances.
    """
    d = torch.cdist(q, q)
    d_self = d.masked_fill(torch.eye(d.shape[0], dtype=torch.bool, device=q.device), float("inf"))
    blocks = {
        "all": d_self,
        "uni": d_self[:n_uni, :n_uni],
        "sharp": d_self[n_uni:, n_uni:],
        "cross": d[:n_uni, n_uni:],
    }
    out = {}
    for name, block in blocks.items():
        nn = block.min(dim=1).values
        out[f"{name}_min"] = float(nn.min())
        out[f"{name}_mean"] = float(nn.mean())
        out[f"{name}_max"] = float(nn.max())
    return out


def summarise(per_shape: list[dict]) -> dict:
    out = {}
    for b in BLOCKS:
        for s in STATS:
            v = np.array([r[f"{b}_{s}"] for r in per_shape], dtype=np.float64)
            out[f"{b}_{s}"] = {
                "mean": float(v.mean()),
                "median": float(np.median(v)),
                "p5": float(np.percentile(v, 5)),
                "p95": float(np.percentile(v, 95)),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16",
                    help="only its preprocess/capacity block is used; both samplers come from it")
    ap.add_argument("--shapes", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json-out", default="results/query_spacing.json")
    ap.add_argument("--md-out", default="docs/writeup/query_spacing.md")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}"])
    pre = instantiate(cfg.preprocess)
    n_uni = pre.random_sample_count

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

    t0 = time.time()
    rows = {"vanilla": [], "disjoint": []}
    for rel, cpath in tqdm(work, unit="shape"):
        try:
            cache = torch.load(cpath, weights_only=False)
            # Same seed before each sampler, so both see the identical downsampled pools and the
            # two methods are compared on one shape rather than on two different random draws.
            for name, fn in (("vanilla", pre.sample), ("disjoint", pre.sample_disjoint)):
                torch.manual_seed(args.seed)
                *_, query_xyz = fn(cache)
                rows[name].append(spacing(query_xyz.to(args.device), n_uni))
        except Exception as e:
            print(f"\n  skip {rel}: {type(e).__name__}: {e}", flush=True)

    if len(rows["vanilla"]) < 2:
        raise SystemExit("not enough shapes measured")
    out = {k: summarise(v) for k, v in rows.items()}
    out["n_shapes"] = len(rows["vanilla"])
    out["seed"] = args.seed
    out["experiment"] = args.experiment
    out["seconds"] = round(time.time() - t0, 1)

    # The ratio column of the README table, stored rather than left to the reader to divide.
    out["ratios"] = {f"{b}_{s}": out["disjoint"][f"{b}_{s}"]["mean"] / out["vanilla"][f"{b}_{s}"]["mean"]
                     for b in BLOCKS for s in STATS}

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.json_out}")

    table = [("Closest pair of query points, any branch", "all_min"),
             ("From each uniform query point to its nearest sharp query point", "cross_min"),
             ("Mean spacing, all query points", "all_mean"),
             ("Mean spacing among sharp query points only", "sharp_mean"),
             ("Mean spacing among uniform query points only", "uni_mean")]
    lines = [f"# Query-point spacing ({out['n_shapes']} shapes of data/{args.split})", "",
             f"- generated by `scripts/query_spacing.py`, seed {args.seed}, "
             f"experiment `{args.experiment}`",
             "- both samplers run on the same shape from the same downsampled pools",
             "- each row is the mean over shapes of a per-shape statistic", "",
             "| Query-point spacing | Original | Two-stage | Ratio |", "|---|---|---|---|"]
    for label, key in table:
        v, d = out["vanilla"][key]["mean"], out["disjoint"][key]["mean"]
        lines.append(f"| {label} | {v:.4f} | {d:.4f} | {d / v:.2f}x |")
    lines += ["", "Full precision, and the min/mean/max of every block with its median and 5/95th "
                  f"percentiles, are in `{args.json_out}`.", ""]
    os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
    with open(args.md_out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.md_out}\n")
    print("\n".join(lines[6:]))


if __name__ == "__main__":
    main()

"""
Nearest-neighbour spacing of the 1,024 query points, under both samplers, over a whole split.

    uv run python -m scripts.query_spacing --shapes 0 --device cuda

Needs no model and no marching cubes -- only the preprocessor. Reproduces the table in the README's
Fix 1 section, including the ratio column, and writes both a JSON and a markdown table.
"""
