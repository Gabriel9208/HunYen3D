from __future__ import annotations

import logging

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.engine.builder import (
    build_dataloaders,
    build_optimizer,
    build_scheduler,
)
from src.engine.checkpoint import CheckpointManager
from src.engine.logging import WandbLogger
from src.engine.utils import (
    count_params,
    get_device,
    move_to_device,
    set_seed,
)

log = logging.getLogger(__name__)


class Trainer:
    """Plain-PyTorch training loop: AMP, grad accumulation, grad clipping,
    per-step scheduler, logging cadence, checkpoint save/resume.

    Depends only on the `Task` contract, never on a specific model.
    """

    def __init__(
        self,
        cfg: DictConfig,  # the `trainer` subconfig
        model,
        task,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        logger: WandbLogger,
        ckpt_manager: CheckpointManager,
        device: torch.device,
    ):
        self.cfg = cfg
        self.model = model
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.ckpt = ckpt_manager
        self.device = device

        self.amp = bool(cfg.amp) and device.type == "cuda"
        self.grad_accum = max(1, int(cfg.grad_accum_steps))
        self.grad_clip = cfg.grad_clip
        self.max_epochs = int(cfg.max_epochs)
        self.log_interval = int(cfg.log_interval)
        self.val_interval = int(cfg.val_interval)
        self.ckpt_interval = int(cfg.ckpt_interval)
        self.progress = bool(cfg.get("progress", True))

        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")

    def fit(self) -> None:
        if self.cfg.get("resume"):
            self._resume(self.cfg.resume)

        with logging_redirect_tqdm():
            epoch_bar = tqdm(
                range(self.epoch, self.max_epochs),
                desc="epochs",
                position=0,
                disable=not self.progress,
            )
            for epoch in epoch_bar:
                self.epoch = epoch
                self._train_epoch()

                do_val = self.val_loader is not None and (epoch + 1) % self.val_interval == 0
                if do_val:
                    metric = self.validate()
                    is_best = metric < self.best_metric
                    self.best_metric = min(metric, self.best_metric)
                    self._save(is_best)
                    epoch_bar.set_postfix(best=self.best_metric)
                elif (epoch + 1) % self.ckpt_interval == 0:
                    self._save(is_best=False)

            self._save(is_best=False)

    def _train_epoch(self) -> None:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            self.train_loader,
            desc=f"epoch {self.epoch}",
            leave=False,
            position=1,
            disable=not self.progress,
        )
        for i, batch in enumerate(pbar):
            batch = move_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                out = self.task.step(self.model, batch)
                loss = out.loss / self.grad_accum
            loss.backward()

            if (i + 1) % self.grad_accum != 0:
                continue

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1

            pbar.set_postfix(
                loss=out.loss.item(),
                lr=self.optimizer.param_groups[0]["lr"],
            )

            if self.global_step % self.log_interval == 0:
                metrics = {
                    "train/loss": out.loss.item(),
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": self.epoch,
                }
                metrics.update({f"train/{k}": v for k, v in out.metrics.items()})
                self.logger.log(metrics, step=self.global_step)
                log.info(
                    "epoch %d step %d loss %.4f",
                    self.epoch,
                    self.global_step,
                    out.loss.item(),
                )

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total, n = 0.0, 0
        sums: dict[str, float] = {}
        val_bar = tqdm(
            self.val_loader,
            desc="val",
            leave=False,
            position=1,
            disable=not self.progress,
        )
        for batch in val_bar:
            batch = move_to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                out = self.task.step(self.model, batch)
            total += out.loss.item()
            n += 1
            for k, v in out.metrics.items():
                sums[k] = sums.get(k, 0.0) + v
        avg = total / max(n, 1)
        metrics = {"val/loss": avg, "epoch": self.epoch}
        metrics.update({f"val/{k}": v / max(n, 1) for k, v in sums.items()})
        self.logger.log(metrics, step=self.global_step)
        log.info("[val] epoch %d loss %.4f", self.epoch, avg)
        return avg

    def _save(self, is_best: bool) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": self.epoch + 1,  # resume starts on the next epoch
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "cfg": OmegaConf.to_container(self.cfg, resolve=True),
        }
        path = self.ckpt.save(state, is_best=is_best)
        log.info("saved checkpoint -> %s", path)

    def _resume(self, path: str) -> None:
        ckpt = CheckpointManager.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_metric = ckpt.get("best_metric", float("inf"))
        log.info("resumed from %s at epoch %d step %d", path, self.epoch, self.global_step)


def run(cfg: DictConfig) -> None:
    tcfg = cfg.trainer
    set_seed(int(tcfg.seed))
    device = get_device(tcfg.get("device", "auto"))
    log.info("device: %s", device)

    model = instantiate(cfg.model).to(device)
    task = instantiate(cfg.task)
    optimizer = build_optimizer(cfg.optimizer, model.parameters())

    # 先建 dataloader 才知道 len(train_loader);scheduler 的 T_max 用「總 optimizer step 數」自動算。
    train_loader, val_loader = build_dataloaders(cfg.data)
    steps_per_epoch = len(train_loader) // max(1, int(tcfg.grad_accum_steps))
    total_steps = steps_per_epoch * int(tcfg.max_epochs)
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer, total_steps=total_steps)
    log.info("scheduler total_steps (T_max if missing): %d", total_steps)

    logger = WandbLogger(cfg.wandb, full_cfg=cfg)
    logger.init()
    logger.watch(model)
    log.info("model params: %s", f"{count_params(model):,}")

    ckpt_manager = CheckpointManager(tcfg.ckpt_dir)

    trainer = Trainer(
        cfg=tcfg,
        model=model,
        task=task,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        ckpt_manager=ckpt_manager,
        device=device,
    )
    try:
        trainer.fit()
    finally:
        logger.finish()
