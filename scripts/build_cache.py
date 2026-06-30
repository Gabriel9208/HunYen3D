"""離線預算前處理快取(訓練前跑一次)。

對 train/val 的所有 .obj 跑 `Preprocessor.build_cache`(heavy:mesh_to_sdf + 表面取樣),
把「大池 + SDF bank」存到 cache_dir/<mesh_stem>.pt,之後訓練走 light 模式只做便宜的子採樣。

用法:
  uv run python scripts/build_cache.py +experiment=overfit        # 複用該實驗的 preprocess/data 參數
  uv run python scripts/build_cache.py +experiment=overfit force=true   # 強制重建

可重現:每個 mesh 以確定 seed(以排序後的索引)取樣;mesh_to_sdf 給定 query 點為確定, 
所以重跑會得到相同快取。
"""

from __future__ import annotations

import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

# 讓 `scripts/` 底下執行也能 import 到 repo 根目錄的 `src`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

from src.engine.utils import set_seed

_PREPROCESSOR = None

def init_worker(cfg: DictConfig):
    global _PREPROCESSOR
    _PREPROCESSOR = instantiate(cfg)

def _build_tasks(obj_root: str, cache_dir: str, force: bool,
                 pattern: str):
    os.makedirs(cache_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(obj_root, "**", pattern), recursive=True))
    if not paths:
        print(f"[build_cache] 警告:{obj_root} 下沒有符合 {pattern!r} 的 mesh,略過")
        return []

    tasks = []
    for idx, p in enumerate(paths):
        # 鏡像 obj_root 的相對路徑(換成 .pt),避免不同子目錄同名(如 ShapeNet hash 重名)互相覆蓋。
        rel = os.path.relpath(p, obj_root)
        out = os.path.join(cache_dir, os.path.splitext(rel)[0] + ".pt")
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
            raise ValueError(f"data.{split}.dataset.cache_dir 未設定")
        # 同一個 (obj_root, cache_dir) 只跑一次
        key = f"{ds.obj_root}->{cache_dir}"
        if key in seen:
            continue
        seen.add(key)
        pattern = ds.get("pattern", "*.obj")  # 與 ObjMeshDataset 同一來源(data config)
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
