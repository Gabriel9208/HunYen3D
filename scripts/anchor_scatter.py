import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)      # runnable as `python scripts/anchor_scatter.py`, not just `-m`
C_UNI, C_SHARP = "#1f77b4", "#d62728"

ap = argparse.ArgumentParser()
ap.add_argument("--shape", default="03001627/4_watertight_scaled/1a6f615e8b1b5ae4dbbc9440457e303e")
ap.add_argument("--run", default="/data/outputs/vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16/"
                                 "2026-09-12_15-43-29", help="only for its preprocess config")
ap.add_argument("--collide", type=float, default=0.02, help="cross-branch distance that counts as a collision")
ap.add_argument("--elev", type=float, default=18)
ap.add_argument("--azim", type=float, default=-60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/anchor_scatter")
a = ap.parse_args()

with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
    cfg = compose(config_name="config", return_hydra_config=True,
                  overrides=[f"+experiment=vae_disjoint_converge_kl1e-4_lr3.5e-6_1024_l8_16"])
HydraConfig.instance().set_config(cfg)
pre = instantiate(cfg.preprocess)
cache = torch.load(os.path.join(ROOT, "cache", a.shape + ".pt"), weights_only=False)
panels = []
for tag, fn in (("independent FPS (vanilla)", pre.sample), ("two-stage FPS (disjoint)", pre.sample_disjoint)):
    torch.manual_seed(a.seed)
    *_, xyz = fn(cache)
    xyz = xyz.numpy()
    u, s = xyz[:512], xyz[512:]
    d_us, i_us = cKDTree(s).query(u)       # each uniform anchor -> nearest sharp anchor
    d_su = cKDTree(u).query(s)[0]
    panels.append((tag, u, s, d_us, d_su, i_us))

# ShapeNet is +Y up but matplotlib draws Z vertical
up = lambda p: (p[:, 0], p[:, 2], p[:, 1])

def style(ax, title):
    ax.set_title(title, fontsize=10.5, pad=-6)
    ax.view_init(elev=a.elev, azim=a.azim)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_box_aspect([1, 1, 1]); ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.02)
        axis.line.set_alpha(0.25)
        axis.set_tick_params(length=0)

fig = plt.figure(figsize=(14.5, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[2.3, 2.3, 1.0], hspace=0.0, wspace=0.0)
cols = ["uniform branch only (512)", "important / sharp branch only (512)", "both + collisions"]

for row, (tag, u, s_, d_us, d_su, i_us) in enumerate(panels):
    m = d_us < a.collide
    for col in range(3):
        ax = fig.add_subplot(gs[row, col], projection="3d")
        if col == 0:
            ax.scatter(*up(u), c=C_UNI, s=10, alpha=0.85, linewidths=0)
        elif col == 1:
            ax.scatter(*up(s_), c=C_SHARP, s=10, alpha=0.85, linewidths=0)
        else:
            ax.scatter(*up(u), c=C_UNI, s=10, alpha=0.7, linewidths=0, label="uniform")
            ax.scatter(*up(s_), c=C_SHARP, s=10, alpha=0.7, linewidths=0, label="important / sharp")
            if m.any():
                ax.scatter(*up(u[m]), facecolors="none", edgecolors="k", s=90, linewidths=1.1,
                           zorder=6, label=f"cross-branch pair < {a.collide}")
        style(ax, cols[col] if row == 0 else "")
        if col == 0:
            ax.text2D(-0.02, 0.5, tag, transform=ax.transAxes, rotation=90, va="center",
                      ha="center", fontsize=12, weight="bold")
        if col == 2:
            ax.text2D(0.5, 0.04, f"closest pair {d_us.min():.4f}   collisions {int(m.sum())}/512",
                      transform=ax.transAxes, ha="center", fontsize=10)
            if row == 0:
                ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=9, framealpha=0.9)

ax = fig.add_subplot(gs[2, :])
bins = np.linspace(0, 0.25, 70)
for (tag, _, _, d_us, _, _), ls, c in zip(panels, ("-", "--"), ("#1f77b4", "#ff7f0e")):
    ax.hist(d_us, bins=bins, histtype="step", lw=1.9, ls=ls, color=c,
            label=f"{tag}   median {np.median(d_us):.4f}")
ax.axvline(a.collide, color="k", lw=0.9, ls=":", label=f"collision threshold {a.collide}")
ax.set_xlabel("distance from each uniform anchor to its nearest sharp anchor", fontsize=11)
ax.set_ylabel("anchors"); ax.legend(fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)

os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
stem = os.path.join(ROOT, a.out, a.shape.split("/")[0] + "_" + a.shape.split("/")[-1][:12])
fig.savefig(stem + ".png", dpi=170, bbox_inches="tight")
fig.savefig(stem + ".pdf", bbox_inches="tight")
print(f"shape {a.shape}")
for tag, u, s, d_us, *_ in panels:
    print(f"  {tag:<28} closest {d_us.min():.4f}  median {np.median(d_us):.4f}  "
          f"collisions(<{a.collide}) {(d_us < a.collide).sum()}/512")
print(f"-> {stem}.png / .pdf")


"""
    uv run python scripts/anchor_scatter.py --shape 03001627/4_watertight_scaled/<id>
"""