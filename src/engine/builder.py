from __future__ import annotations

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

"""Thin wrappers over Hydra's `instantiate` so construction stays config-driven
and in one place. Optimizer/scheduler configs use `_partial_: true`, so
instantiate returns a partial that we bind to params/optimizer here."""


def build_model(cfg: DictConfig):
    return instantiate(cfg)


def build_task(cfg: DictConfig):
    return instantiate(cfg)


def build_optimizer(cfg: DictConfig, params):
    return instantiate(cfg)(params)


def build_scheduler(cfg: DictConfig | None, optimizer, total_steps: int | None = None):
    if cfg is None:
        return None
    # 若 config 把 T_max 留成 ???(missing),用 total_steps 自動補(= len(train_loader) × epochs);
    # 使用者有明確設定 T_max 時則尊重該值;沒有 T_max 這個 key 的 scheduler 則略過。
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
