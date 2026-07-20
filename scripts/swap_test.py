"""Swap test: diagnose posterior collapse (whether the decoder actually uses the latent).

Encode shapes A and B, take mu each (use the mode, no sampling, drop noise), and compare
reconstructing A with the "right" vs "wrong" latent:
  decode(z_A, qp_A) vs decode(z_B, qp_A)
If swapping in the wrong latent z_B barely worsens the reconstruction of A → the decoder isn't
using the latent = collapse.
Also reports per-element KL and the active-units fraction (share of latent scalars with KL>threshold),
a direct read of whether it collapsed.

Usage:
  uv run python scripts/swap_test.py +experiment=base_small \
      +ckpt=outputs/2026-07-02/00-58-58/checkpoints/last.pt
  # optional: +pairs=8
"""

from __future__ import annotations

import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.engine.utils import get_device, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(0)
    device = get_device(cfg.trainer.get("device", "auto"))
    model = instantiate(cfg.model).to(device).eval()
    state = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    ds = instantiate(cfg.data.val.dataset)
    n_pairs = int(cfg.get("pairs", 8))

    @torch.no_grad()
    def encode_mu(item):
        q = item["query"].unsqueeze(0).to(device)
        d = item["data"].unsqueeze(0).to(device)
        return model.encoder(q, d)  # (mu, logvar)

    recon, swap, signflip, kl_mean, active = [], [], [], [], []
    with torch.no_grad():
        for i in range(n_pairs):
            a, b = ds[2 * i], ds[2 * i + 1]
            mu_a, logvar_a = encode_mu(a)
            mu_b, _ = encode_mu(b)
            qp = a["query_points"].unsqueeze(0).to(device)
            gt = a["gt_sdf"].unsqueeze(0).to(device)

            sdf_aa = model.decode(mu_a, qp)[0]   # right latent (decode → (sdf, anchors))
            sdf_ba = model.decode(mu_b, qp)[0]   # wrong latent (B's) reconstructing A

            recon.append(F.mse_loss(sdf_aa, gt).item())
            swap.append(F.mse_loss(sdf_ba, gt).item())
            signflip.append((torch.sign(sdf_aa) != torch.sign(sdf_ba)).float().mean().item())

            kl_elem = 0.5 * (mu_a.pow(2) + logvar_a.exp() - 1.0 - logvar_a)  # (1, L, D)
            kl_mean.append(kl_elem.mean().item())
            active.append((kl_elem > 1e-2).float().mean().item())

    r, s = st.mean(recon), st.mean(swap)
    print(f"\npairs={n_pairs}  ckpt={cfg.ckpt}")
    print(f"recon MSE (z_A→A)      : {r:.5f}")
    print(f"swap  MSE (z_B→A)      : {s:.5f}")
    print(f"swap/recon ratio       : {s / max(r, 1e-9):.2f}x   (>>1 healthy; ≈1 = decoder ignores latent = collapse)")
    print(f"sign-flip on swap       : {st.mean(signflip) * 100:.1f}%   (how many query points flip inside/outside with wrong latent; ≈0% = collapse)")
    print(f"mean per-elem KL       : {st.mean(kl_mean):.5f}")
    print(f"active units (KL>1e-2) : {st.mean(active) * 100:.1f}%   (lower = more collapsed)")


if __name__ == "__main__":
    main()
