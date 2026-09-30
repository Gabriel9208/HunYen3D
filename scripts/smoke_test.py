from __future__ import annotations

import os
import sys

# So running under scripts/ can still import `src` from the repo root
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
        assert torch.isfinite(out.loss), f"step {i}: loss is not finite {out.loss}"
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in out.metrics.items())
        print(f"step {i}: loss={out.loss.item():.4f} {metrics_str}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[smoke] OK — {steps} steps no OOM, peak VRAM {peak:.2f} / {total:.2f} GiB")


if __name__ == "__main__":
    main()

"""
  uv run python scripts/smoke_test.py +experiment=base_first_train
  uv run python scripts/smoke_test.py +experiment=base_first_train steps=5
"""