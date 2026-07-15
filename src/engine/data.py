from __future__ import annotations

import glob
import os

import torch
from torch.utils.data import Dataset

from src.engine.utils import cache_rel_path, set_seed

class ObjMeshDataset(Dataset):
    """Turns .obj meshes into VAE inputs via `Preprocessor`. Handles IO + mode +
    reproducibility; the conversion logic itself lives in `Preprocessor` (single source of truth).

    Two modes:
    - mode="light" (default): read `cache_dir/<mesh_stem>.pt` (the big pool / SDF bank produced
      offline by build_cache) → `preprocessor.sample(cache)`. Each epoch only does the cheap
      subsampling + FPS + Fourier.
    - mode="heavy": run the full preprocessing on the fly via `preprocessor(mesh_path)`
      (slow, for debug / when there is no cache).

    When `fixed_seed` is not None, `set_seed(fixed_seed+idx)` runs before sampling each item →
    identical every epoch (used to freeze overfit); None re-samples each epoch (training variety).
    """

    def __init__(self, obj_root: str, preprocessor, pattern: str = "*.obj",
                 mode: str = "light", cache_dir: str | None = None,
                 fixed_seed: int | None = None, max_meshes: int | None = None):
        self.paths = sorted(
            glob.glob(os.path.join(obj_root, "**", pattern), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"No meshes matching {pattern!r} under {obj_root!r}")
        # max_meshes truncates the (sorted → deterministic) list, e.g. =1 for single-mesh overfit:
        # the same shape every run and across configs, reusing the existing cache (no separate data dir).
        if max_meshes is not None:
            self.paths = self.paths[:max_meshes]
        self.obj_root = obj_root
        self.preprocessor = preprocessor
        self.mode = mode
        self.cache_dir = cache_dir
        self.fixed_seed = fixed_seed

        if mode not in ("light", "heavy"):
            raise ValueError(f"mode must be 'light' or 'heavy', got {mode!r}")
        if mode == "light" and not cache_dir:
            raise ValueError(
                "light mode needs cache_dir; run scripts/build_cache.py to build the cache first, "
                "or switch to mode=heavy."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_path(self, mesh_path: str) -> str:
        # Same as scripts/build_cache.py: mirror obj_root's relative path so same-named meshes don't collide on one cache file.
        return os.path.join(self.cache_dir, cache_rel_path(os.path.relpath(mesh_path, self.obj_root)))

    def __getitem__(self, idx: int):
        if self.fixed_seed is not None:
            set_seed(self.fixed_seed + idx)

        path = self.paths[idx]
        if self.mode == "light":
            cpath = self._cache_path(path)
            if os.path.exists(cpath):
                cache = torch.load(cpath, weights_only=False)
                self._check_signature(cache, cpath)
            else:  # cache missing → rebuild it heavy from the mesh, next time falls back to light
                cache = self.preprocessor.build_cache(path)
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                torch.save(cache, cpath)
            q, d, query_points, gt_sdf = self.preprocessor.sample(cache)
        else:  # heavy mode (debug): run on the fly, no cache saved
            q, d, query_points, gt_sdf = self.preprocessor(path)

        return {
            "query": q,                    # (L_q, encoder_pe_dim)
            "data": d,                     # (L_d, encoder_pe_dim)
            "query_points": query_points,  # (S, decoder_pe_dim)
            "gt_sdf": gt_sdf,              # (S, 1)
        }

    def _check_signature(self, cache: dict, cpath: str) -> None:
        sig = cache.get("signature")
        want = self.preprocessor.cache_signature
        if sig != want:
            raise ValueError(
                f"cache parameter signature mismatch ({cpath}): cache={sig} vs config={want}; "
                f"rebuild with `scripts/build_cache.py ... force=true`."
            )
