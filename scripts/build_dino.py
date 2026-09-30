#!/usr/bin/env python3
"""Precompute DINOv3-Large patch-token features for the rendered views into a cond cache,
keyed exactly like the mesh cache so MeshCondDataset (src/engine/data.py) can load them.

Per model: load all rendered views -> DINOv3-L patch tokens -> (V, num_patches, 1024) fp16 ->
save to <out>/<cat>/4_watertight_scaled/<id>.pt.

Setup (weights are gated — accept the license on HF, then `huggingface-cli login`):
    uv add transformers pillow

Run (GPU; renders must exist under --render-root):
    uv run python scripts/build_dino.py --render-root data/rendered --out /data/cache_dino
    uv run python scripts/build_dino.py --dry ...     # pipeline check, no model/weights needed
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm

MESH_SUBDIR = "4_watertight_scaled"     # cond key mirrors the mesh cache (cache_rel_path)


def load_rgb(path: str):
    from PIL import Image
    im = Image.open(path).convert("RGBA")                     # renders are transparent
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))     # composite on white (natural bg)
    return Image.alpha_composite(bg, im).convert("RGB")


def load_dino(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    return model, proc


@torch.no_grad()
def extract(model, proc, images, device) -> torch.Tensor:
    """images: list[PIL] (V views) -> (V, num_patches, D) fp16 patch tokens."""
    inputs = proc(images=images, return_tensors="pt").to(device)
    out = model(**inputs).last_hidden_state                   # (V, T, D)  T = CLS + registers + patches
    h, w = inputs["pixel_values"].shape[-2:]
    n_patch = (h // model.config.patch_size) * (w // model.config.patch_size)
    return out[:, -n_patch:].half().cpu()                     # patches are the trailing tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-root", default="data/rendered")
    ap.add_argument("--out", default="/data/cache_dino")
    ap.add_argument("--model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="cap models (smoke-test the model before the full run)")
    ap.add_argument("--dry", action="store_true", help="random features, no model (pipeline test)")
    args = ap.parse_args()

    out_root = Path(args.out)
    render_dirs = sorted(glob.glob(os.path.join(args.render_root, "*", "*", "*", "rendering")))
    if not render_dirs:
        raise SystemExit(f"no render dirs under {args.render_root}/<split>/<cat>/<id>/rendering")
    if args.limit:
        render_dirs = render_dirs[:args.limit]

    model = proc = None
    if not args.dry:
        model, proc = load_dino(args.model_id, args.device)

    done = skipped = 0
    for d in tqdm(render_dirs, unit="model", desc="dino"):
        parts = Path(d).parts                                 # .../<split>/<cat>/<id>/rendering
        cat, mid = parts[-3], parts[-2]
        out = out_root / cat / MESH_SUBDIR / f"{mid}.pt"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        pngs = sorted(glob.glob(os.path.join(d, "*.png")))
        if not pngs:
            continue
        imgs = [load_rgb(p) for p in pngs]
        feats = (torch.randn(len(imgs), 196, 1024, dtype=torch.float16) if args.dry
                 else extract(model, proc, imgs, args.device))
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats.contiguous(), out)                   # (V, num_patches, 1024) fp16
        done += 1
    print(f"dino cache: wrote {done}, skipped {skipped} -> {out_root}")


if __name__ == "__main__":
    main()
