from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.engine.utils import is_main_process


class WandbLogger:
    """Thin wrapper around Weights & Biases.

    Honors `cfg.mode`: when "disabled" (the default for smoke tests) nothing is
    sent and wandb is never even imported. "offline" writes a local run dir;
    "online" logs to the server.
    """

    def __init__(self, cfg: DictConfig, full_cfg: DictConfig | None = None):
        self.cfg = cfg
        self.full_cfg = full_cfg
        self.enabled = is_main_process() and cfg.get("mode", "disabled") != "disabled"
        self._run = None

    def init(self) -> None:
        if not self.enabled:
            return
        import wandb

        self._run = wandb.init(
            project=self.cfg.get("project"),
            entity=self.cfg.get("entity"),
            name=self.cfg.get("name"),
            tags=list(self.cfg.get("tags", []) or []),
            mode=self.cfg.get("mode", "online"),
            config=(
                OmegaConf.to_container(self.full_cfg, resolve=True)
                if self.full_cfg is not None
                else None
            ),
        )

    def watch(self, model) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.watch(model)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.enabled and self._run is not None:
            import wandb

            wandb.finish()
