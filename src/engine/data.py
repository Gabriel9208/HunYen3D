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
                 fixed_seed: int | None = None, max_meshes: int | None = None,
                 categories: list[str] | None = None):
        self.paths = sorted(
            glob.glob(os.path.join(obj_root, "**", pattern), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"No meshes matching {pattern!r} under {obj_root!r}")
        # categories restricts to specific top-level category folders under obj_root (e.g. ShapeNet
        # synset IDs), for category-subset experiments without touching the shared data/ tree.
        if categories is not None:
            self.paths = [p for p in self.paths if os.path.relpath(p, obj_root).split(os.sep)[0] in categories]
            if not self.paths:
                raise FileNotFoundError(f"No meshes for categories={categories!r} under {obj_root!r}")
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
            q, d, query_points, gt_sdf, query_xyz = self._sample(cache)
        else:  # heavy mode (debug): run on the fly, no cache saved
            q, d, query_points, gt_sdf, query_xyz = self.preprocessor(path)

        return {
            "query": q,                    # (L_q, encoder_pe_dim)
            "data": d,                     # (L_d, encoder_pe_dim)
            "query_points": query_points,  # (S, decoder_pe_dim)
            "gt_sdf": gt_sdf,              # (S, 1)
            "query_xyz": query_xyz,        # (num_latents, 3) FPS query coords, anchor GT
        }

    def _sample(self, cache: dict):
        return self.preprocessor.sample(cache)

    def _check_signature(self, cache: dict, cpath: str) -> None:
        sig = cache.get("signature")
        want = self.preprocessor.cache_signature
        if sig != want:
            raise ValueError(
                f"cache parameter signature mismatch ({cpath}): cache={sig} vs config={want}; "
                f"rebuild with `scripts/build_cache.py ... force=true`."
            )


class DisjointVAEDataset(ObjMeshDataset):
    def _sample(self, cache: dict):
        return self.preprocessor.sample_disjoint(cache)


class MeshCondDataset(ObjMeshDataset):
    def __init__(self, cond_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.cond_dir = cond_dir

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)  # query, data, query_points, gt_sdf, query_xyz (+ fixed_seed set)
        key = cache_rel_path(os.path.relpath(self.paths[idx], self.obj_root))
        cond = torch.load(os.path.join(self.cond_dir, key), weights_only=True)  # (V, L, D)
        item["cond"] = cond[torch.randint(len(cond), ()).item()]                 # random view
        return item


class MeshCondDisjointDataset(DisjointVAEDataset):
    """DisjointVAEDataset + precomputed DINOv2 cond features, for DiT training on a frozen VAE that
    was itself trained on sample_disjoint() (same cond-loading logic as MeshCondDataset)."""

    def __init__(self, cond_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.cond_dir = cond_dir

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        key = cache_rel_path(os.path.relpath(self.paths[idx], self.obj_root))
        cond = torch.load(os.path.join(self.cond_dir, key), weights_only=True)
        item["cond"] = cond[torch.randint(len(cond), ()).item()]
        return item


class DoubleStreamVAEDataset(Dataset):
    def __init__(self, obj_root: str, preprocessor, pattern: str = "*.obj",
                 mode: str = "light", cache_dir: str | None = None,
                 fixed_seed: int | None = None, max_meshes: int | None = None,
                 categories: list[str] | None = None):
        self.paths = sorted(
            glob.glob(os.path.join(obj_root, "**", pattern), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"No meshes matching {pattern!r} under {obj_root!r}")
        # categories restricts to specific top-level category folders under obj_root (e.g. ShapeNet
        # synset IDs), for category-subset experiments without touching the shared data/ tree.
        if categories is not None:
            self.paths = [p for p in self.paths if os.path.relpath(p, obj_root).split(os.sep)[0] in categories]
            if not self.paths:
                raise FileNotFoundError(f"No meshes for categories={categories!r} under {obj_root!r}")
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
            q_xyz, d_xyz, q_normal, d_normal, q_emb, d_emb, sdf_query_points, gt_sdf = self._sample(cache)
        else:  # heavy mode (debug): run on the fly, no cache saved
            raise RuntimeError("Heavy mode is not implemented")

        return {
            "q_xyz": q_xyz,
            "d_xyz": d_xyz,
            "q_normal": q_normal,
            "d_normal": d_normal,
            "q_emb": q_emb,
            "d_emb": d_emb,
            "query_points": sdf_query_points,   # same key as ObjMeshDataset so compute_loss is shared
            "gt_sdf": gt_sdf,
        }

    def _sample(self, cache: dict):
        return self.preprocessor.double_stream_sample(cache)

    def _check_signature(self, cache: dict, cpath: str) -> None:
        sig = cache.get("signature")
        want = self.preprocessor.cache_signature
        if sig != want:
            raise ValueError(
                f"cache parameter signature mismatch ({cpath}): cache={sig} vs config={want}; "
                f"rebuild with `scripts/build_cache.py ... force=true`."
            )


class FrameVAEDataset(DoubleStreamVAEDataset):
    """Same as DoubleStreamVAEDataset, but the query is 100% importance points (FrameEncoder's
    design never uses a uniform-branch query) -- preprocessor.frame_sample() skips computing the
    uniform branch's query FPS entirely instead of computing it and discarding it downstream."""

    def _sample(self, cache: dict):
        return self.preprocessor.frame_sample(cache)


class MeshCondFrameDataset(FrameVAEDataset):
    """FrameVAEDataset + precomputed DINOv2 cond features, for DiT training on a frozen FrameVAE
    (same cond-loading logic as MeshCondDataset, just wrapping the frame_sample batch keys instead
    of ObjMeshDataset's plain query/data keys)."""

    def __init__(self, cond_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.cond_dir = cond_dir

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        key = cache_rel_path(os.path.relpath(self.paths[idx], self.obj_root))
        cond = torch.load(os.path.join(self.cond_dir, key), weights_only=True)
        item["cond"] = cond[torch.randint(len(cond), ()).item()]  # random view
        return item