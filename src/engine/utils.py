from __future__ import annotations

import os
import random

import numpy as np
import torch


def cache_rel_path(mesh_rel: str) -> str:
    """mesh 相對路徑 → 對應 .pt 相對路徑;剝掉 .gz 與 mesh 副檔名,讓 x.off / x.off.gz 都對到 x.pt。
    build_cache.py / data.py / compress_cached_meshes.py 三處共用,避免命名漂移。"""
    if mesh_rel.endswith(".gz"):
        mesh_rel = mesh_rel[:-3]
    return os.path.splitext(mesh_rel)[0] + ".pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def move_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch
