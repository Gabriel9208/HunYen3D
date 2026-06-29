from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.engine.utils import set_seed

class DummyShapeDataset(Dataset):
    """Random tensors with a learnable target, so the scaffold's loss can actually
    decrease. Replace with the real shape/SDF dataset later."""

    def __init__(self, num_samples: int = 256, dim: int = 16, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(num_samples, dim, generator=g)
        self.targets = torch.tanh(self.inputs) * 2.0

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int):
        return {"input": self.inputs[idx], "target": self.targets[idx]}


class ObjMeshDataset(Dataset):
    """把 .obj mesh 經由 `Preprocessor` 變成 VAE 的輸入。負責 IO + 模式 + 可重現,
    轉換邏輯本身在 `Preprocessor`(單一真相)。

    兩種模式:
    - mode="light"(預設):讀 `cache_dir/<mesh_stem>.pt`(離線 build_cache 產生的大池/SDF bank)
      → `preprocessor.sample(cache)`。每個 epoch 只做便宜的子採樣 + FPS + Fourier。
    - mode="heavy":直接 `preprocessor(mesh_path)` 即時跑完整前處理(慢,debug / 無快取時用)。

    `fixed_seed` 不為 None 時,每個 item 取樣前 `set_seed(fixed_seed+idx)` → 每個 epoch 相同
    (overfit 凍結用);None 則每個 epoch 重抽(訓練的變化性)。
    """

    def __init__(self, obj_root: str, preprocessor, pattern: str = "*.obj",
                 mode: str = "light", cache_dir: str | None = None,
                 fixed_seed: int | None = None):
        self.paths = sorted(
            glob.glob(os.path.join(obj_root, "**", pattern), recursive=True)
        )
        if not self.paths:
            raise FileNotFoundError(f"No meshes matching {pattern!r} under {obj_root!r}")
        self.preprocessor = preprocessor
        self.mode = mode
        self.cache_dir = cache_dir
        self.fixed_seed = fixed_seed

        if mode not in ("light", "heavy"):
            raise ValueError(f"mode 必須是 'light' 或 'heavy',收到 {mode!r}")
        if mode == "light" and not cache_dir:
            raise ValueError(
                "light 模式需要 cache_dir;請先跑 scripts/build_cache.py 建立快取,"
                "或改用 mode=heavy。"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def _cache_path(self, mesh_path: str) -> str:
        return os.path.join(self.cache_dir, Path(mesh_path).stem + ".pt")

    def __getitem__(self, idx: int):
        if self.fixed_seed is not None:
            set_seed(self.fixed_seed + idx)

        if self.mode == "light":
            cpath = self._cache_path(self.paths[idx])
            if not os.path.exists(cpath):
                raise FileNotFoundError(
                    f"找不到快取 {cpath};請先跑 `scripts/build_cache.py` 建立快取"
                    f"(或改用 mode=heavy)。"
                )
            cache = torch.load(cpath, weights_only=False)
            self._check_signature(cache, cpath)
            q, d, query_points, gt_sdf = self.preprocessor.sample(cache)
        else:  # heavy
            q, d, query_points, gt_sdf = self.preprocessor(self.paths[idx])

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
                f"快取參數簽章不符 ({cpath}):cache={sig} vs config={want};"
                f"請用 `scripts/build_cache.py ... force=true` 重建。"
            )
