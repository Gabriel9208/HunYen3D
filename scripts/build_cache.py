"""Offline preprocessing cache (run once before training).

Runs `Preprocessor.build_cache` (heavy: mesh_to_sdf + surface sampling) over all .obj in
train/val, saving the big pool + SDF bank to cache_dir/<mesh_stem>.pt; training then uses light
mode and only does the cheap subsampling.

Usage:
  uv run python scripts/build_cache.py +experiment=overfit        # reuse that experiment's preprocess/data params
  uv run python scripts/build_cache.py +experiment=overfit force=true   # force rebuild

Reproducible: each mesh is sampled with a deterministic seed (its sorted index); mesh_to_sdf is
deterministic for given query points, so re-running yields the same cache.
"""

from __future__ import annotations

import ctypes
import glob
import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor

# So running under scripts/ can still import `src` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.engine.utils import cache_rel_path, set_seed

_PREPROCESSOR = None

def _die_with_parent() -> None:
    # ponytail: Linux-only. When the main process dies (incl. terminal SIGHUP, crash, SIGKILL),
    # the kernel immediately sends this worker SIGKILL, so it won't become a PPID=1 orphan that
    # keeps polluting the cache.
    # Ceiling: if the parent dies before prctl (a tiny startup window) it isn't caught; cross-platform would need psutil polling.
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")


def init_worker(cfg: DictConfig):
    global _PREPROCESSOR
    _die_with_parent()
    _PREPROCESSOR = instantiate(cfg)

def _build_tasks(obj_root: str, cache_dir: str, force: bool,
                 pattern: str):
    os.makedirs(cache_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(obj_root, "**", pattern), recursive=True))
    if not paths:
        print(f"[build_cache] warning: no mesh matching {pattern!r} under {obj_root}, skipping")
        return []

    tasks = []
    for idx, p in enumerate(paths):
        # Mirror obj_root's relative path (as .pt) so same-named meshes in different subdirs (e.g. ShapeNet hash collisions) don't overwrite each other.
        out = os.path.join(cache_dir, cache_rel_path(os.path.relpath(p, obj_root)))
        tasks.append((idx, p, out, force))
    return tasks


def _worker(task: tuple) -> None:
    global _PREPROCESSOR
    idx, p, out, force = task
    if os.path.exists(out) and not force:
        return
    set_seed(idx)  
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cache = _PREPROCESSOR.build_cache(p)
    torch.save(cache, out)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    force = bool(cfg.get("force", False))

    seen: set[str] = set()
    for split in ("train", "val"):
        print(f"[build_cache] start to build {split} preprocess cache...")
        ds = cfg.data[split].dataset
        cache_dir = ds.get("cache_dir")
        if cache_dir is None:
            raise ValueError(f"data.{split}.dataset.cache_dir is not set")
        # Run each (obj_root, cache_dir) only once
        key = f"{ds.obj_root}->{cache_dir}"
        if key in seen:
            continue
        seen.add(key)
        pattern = ds.get("pattern", "*.obj")  # same source as ObjMeshDataset (data config)
        tasks = _build_tasks(ds.obj_root, cache_dir, force, pattern)
        with ProcessPoolExecutor(
            max_workers=8,
            initializer=init_worker,
            initargs=(cfg.preprocess,),
        ) as ex:
            for _ in tqdm(ex.map(_worker, tasks, chunksize=1), total=len(tasks)):
                pass

    print(f"[build_cache] build {split} preprocess cache done")



if __name__ == "__main__":
    main()
