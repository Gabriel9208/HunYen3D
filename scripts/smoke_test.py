"""Smoke test:跑幾個真實 train step,確認不 OOM、loss 有限,並印峰值顯存。

打的是完整訓練路徑(encode → decode → backward → optimizer.step),用真實 cache 資料,
所以能直接驗證 sdf_subset / bf16 這類記憶體設定是否進得去。

用法:
  uv run python scripts/smoke_test.py +experiment=first_train
  uv run python scripts/smoke_test.py +experiment=first_train steps=5
"""

from __future__ import annotations

import os
import sys

# 讓 `scripts/` 底下執行也能 import 到 repo 根目錄的 `src`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.engine.builder import build_dataloaders, build_optimizer
from src.engine.utils import get_device, move_to_device, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    steps = int(cfg.get("steps", 3))
    set_seed(int(cfg.trainer.seed))
    device = get_device(cfg.trainer.get("device", "auto"))
    amp = bool(cfg.trainer.amp) and device.type == "cuda"

    model = instantiate(cfg.model).to(device)
    task = instantiate(cfg.task)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())
    train_loader, _ = build_dataloaders(cfg.data)

    model.train()
    it = iter(train_loader)
    for i in range(steps):
        batch = move_to_device(next(it), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = task.step(model, batch)
        out.loss.backward()
        optimizer.step()
        assert torch.isfinite(out.loss), f"step {i}: loss 非有限值 {out.loss}"
        print(f"step {i}: loss={out.loss.item():.4f} "
              f"recon={out.metrics['recon']:.4f} kl={out.metrics['kl']:.4f}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[smoke] OK — {steps} steps 無 OOM,峰值顯存 {peak:.2f} / {total:.2f} GiB")


if __name__ == "__main__":
    main()
