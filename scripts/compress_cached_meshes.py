"""Compress in place the original meshes that are already validly cached (.off -> .off.gz), to reclaim disk.

Once a mesh's cache_dir/<rel path>.pt exists and loads, training's mode="light" never reads the
original mesh again (see src/engine/data.py ObjMeshDataset.__getitem__), so cached originals can be
safely compressed (reversible, not deleted).

Usage:
  uv run python scripts/compress_cached_meshes.py +experiment=base_first_train dry_run=true  # preview
  uv run python scripts/compress_cached_meshes.py +experiment=base_first_train               # actually compress

Note: after compression, build_cache.py's `*.off` glob won't match `.off.gz`; if you later rebuild
the cache with different heavy params (force=true), the compressed items are silently skipped and
need a manual gunzip first.
"""

from __future__ import annotations

import glob
import gzip
import os
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

# So running under scripts/ can still import `src` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.engine.utils import cache_rel_path

_REQUIRED_KEYS = {"surface_pool", "sharp_pool", "sdf_query_points", "gt_sdf", "signature"}


def _cache_path(mesh_path: str, obj_root: str, cache_dir: str) -> str:
    # Same as scripts/build_cache.py and src/engine/data.py: mirror obj_root's relative path.
    return os.path.join(cache_dir, cache_rel_path(os.path.relpath(mesh_path, obj_root)))


def _is_valid_cache(cpath: str) -> bool:
    try:
        cache = torch.load(cpath, weights_only=False)
    except Exception:
        return False
    return _REQUIRED_KEYS.issubset(cache.keys())


def _build_tasks(obj_root: str, cache_dir: str, pattern: str):
    paths = sorted(glob.glob(os.path.join(obj_root, "**", pattern), recursive=True))
    return [(p, _cache_path(p, obj_root, cache_dir)) for p in paths]


def _classify(mesh_path: str, cache_path: str) -> str:
    gz_path = mesh_path + ".gz"
    if os.path.exists(gz_path) and not os.path.exists(mesh_path):
        return "already_done"
    if not os.path.exists(cache_path):
        return "cache_missing"
    if not _is_valid_cache(cache_path):
        return "cache_invalid"
    return "eligible"


def _worker(task: tuple) -> dict:
    mesh_path, cache_path = task
    status = _classify(mesh_path, cache_path)
    if status != "eligible":
        return {"status": status, "orig_size": 0, "comp_size": 0}

    gz_path = mesh_path + ".gz"
    tmp_path = gz_path + ".tmp"
    orig_size = os.path.getsize(mesh_path)

    with open(mesh_path, "rb") as fin, gzip.open(tmp_path, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)

    decompressed_size = 0
    with gzip.open(tmp_path, "rb") as fcheck:
        for chunk in iter(lambda: fcheck.read(1024 * 1024), b""):
            decompressed_size += len(chunk)

    if decompressed_size != orig_size:
        os.remove(tmp_path)
        return {"status": "verify_failed", "orig_size": 0, "comp_size": 0}

    os.replace(tmp_path, gz_path)
    os.remove(mesh_path)
    return {"status": "compressed", "orig_size": orig_size, "comp_size": os.path.getsize(gz_path)}


def _print_summary(counts: Counter, total_orig: int, total_comp: int) -> None:
    print("[compress_cached_meshes] summary:")
    for status, n in counts.items():
        print(f"  {status}: {n}")
    gb = 1024 ** 3
    print(f"[compress_cached_meshes] original size: {total_orig / gb:.2f} G, "
          f"compressed: {total_comp / gb:.2f} G, reclaimed: {(total_orig - total_comp) / gb:.2f} G")
    print("[compress_cached_meshes] warning: if you later rebuild the cache for a compressed mesh with "
          "different heavy params (force=true), build_cache.py's `*.off` glob won't match `.off.gz` "
          "and it is silently skipped; manually gunzip the file first if needed.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dry_run = bool(cfg.get("dry_run", False))

    seen: set[str] = set()
    counts: Counter = Counter()
    total_orig = 0
    total_comp = 0

    for split in ("train", "val", "test"):
        ds = cfg.data[split].dataset
        cache_dir = ds.get("cache_dir")
        if cache_dir is None:
            raise ValueError(f"data.{split}.dataset.cache_dir is not set")
        key = f"{ds.obj_root}->{cache_dir}"
        if key in seen:
            continue
        seen.add(key)

        pattern = ds.get("pattern", "*.obj")
        tasks = _build_tasks(ds.obj_root, cache_dir, pattern)
        print(f"[compress_cached_meshes] {split}: {len(tasks)} candidate meshes")

        if dry_run:
            for mesh_path, cache_path in tqdm(tasks):
                status = _classify(mesh_path, cache_path)
                counts[status] += 1
                if status == "eligible":
                    total_orig += os.path.getsize(mesh_path)
            continue

        with ProcessPoolExecutor(max_workers=8) as ex:
            for result in tqdm(ex.map(_worker, tasks, chunksize=1), total=len(tasks)):
                counts[result["status"]] += 1
                total_orig += result["orig_size"]
                total_comp += result["comp_size"]

    _print_summary(counts, total_orig, total_comp)


if __name__ == "__main__":
    main()
