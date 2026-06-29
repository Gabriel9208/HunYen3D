from __future__ import annotations

from pathlib import Path

import torch

from src.engine.utils import is_main_process


class CheckpointManager:
    """Saves `last.pt` every checkpoint and `best.pt` when a new best metric is hit.

    Only the main process writes (seam for future DDP).
    """

    def __init__(self, ckpt_dir: str):
        self.ckpt_dir = Path(ckpt_dir)
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: dict, is_best: bool = False) -> Path | None:
        if not is_main_process():
            return None
        last_path = self.ckpt_dir / "last.pt"
        torch.save(state, last_path)
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
        return last_path

    @staticmethod
    def load(path, map_location="cpu") -> dict:
        return torch.load(path, map_location=map_location, weights_only=False)
