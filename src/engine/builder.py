from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

"""Thin wrappers over Hydra's `instantiate` so construction stays config-driven
and in one place. Optimizer/scheduler configs use `_partial_: true`, so
instantiate returns a partial that we bind to params/optimizer here."""


def build_optimizer(cfg: DictConfig, params):
    return instantiate(cfg)(params)


def build_scheduler(cfg: DictConfig | None, optimizer, total_steps: int | None = None):
    if cfg is None:
        return None
    # If the config leaves T_max as ??? (missing), fill it from total_steps (= len(train_loader) × epochs);
    # respect an explicitly set T_max; skip schedulers that have no T_max key at all.
    kwargs = {}
    if total_steps is not None:
        try:
            t_max_missing = OmegaConf.is_missing(cfg, "T_max")
        except Exception:
            t_max_missing = False
        if t_max_missing:
            kwargs["T_max"] = total_steps
    return instantiate(cfg)(optimizer, **kwargs)


def build_dataloaders(cfg: DictConfig):
    train_ds = instantiate(cfg.train.dataset)
    val_ds = instantiate(cfg.val.dataset)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader
