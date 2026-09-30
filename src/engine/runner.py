from __future__ import annotations

import itertools
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

# lambda-VAE schedule buffers (registered persistent=False on VAE, see vae.py) -- saved
# and restored explicitly so a resume continues the ramp instead of starting it over.
_LAM_BUFFERS = ("sigma_ema", "sigma_acc", "acc_n", "lam_star", "lam_step")


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
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

        # best.pt selection metric + direction (config-driven so VAE keeps val/loss+min while DiT uses val/viou_mean+max)
        # Non-finite guard: skip the offending micro-batch, abort once nothing finite comes out.
        self.nonfinite_patience = int(cfg.get("nonfinite_patience", 50))
        self.nonfinite_streak = 0
        self.nonfinite_total = 0
        self.best_key = cfg.get("best_metric_key", "val/loss")
        self.best_mode = cfg.get("best_metric_mode", "min")

        self.epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if self.best_mode == "min" else float("-inf")

    def fit(self) -> None:
        if self.cfg.get("resume"):
            self._resume(self.cfg.resume)
        elif self.cfg.get("init_from"):
            # Warm start: model weights only. Optimizer moments, epoch counter and best_metric all stay
            # fresh, so this is a NEW run that happens to start from a trained model -- not a resume.
            ckpt = CheckpointManager.load(self.cfg.init_from, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            log.info("warm start from %s (weights only, epoch %s)", self.cfg.init_from, ckpt.get("epoch"))

        with logging_redirect_tqdm():
            # max_epochs < 0 (sentinel -1) -> train forever until the process is stopped;
            # last.pt is saved every epoch so a manual kill loses at most the current epoch.
            epochs = (
                itertools.count(self.epoch)
                if self.max_epochs < 0
                else range(self.epoch, self.max_epochs)
            )
            epoch_bar = tqdm(
                epochs,
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
                    if self.best_mode == "min":
                        is_best = metric < self.best_metric
                        self.best_metric = min(metric, self.best_metric)
                    else:
                        is_best = metric > self.best_metric
                        self.best_metric = max(metric, self.best_metric)
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

            # Non-finite guard, forward side. Dropping the micro-batch before backward() keeps the
            # weights clean -- a NaN loss has not touched them yet, but backprop through it would.
            if not torch.isfinite(loss):
                self._nonfinite("loss", out, batch)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss.backward()

            if (i + 1) % self.grad_accum != 0:
                continue

            # Non-finite guard, backward side. grad_clip already reduces over every gradient, so its
            # returned norm is a free NaN/Inf detector -- and gradients can be non-finite even when
            # every loss in the window was finite (0 * inf inside a masked branch, for one).
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip) \
                if self.grad_clip else None
            if grad_norm is not None and not torch.isfinite(grad_norm):
                self._nonfinite("grad_norm", out, batch, grad_norm=grad_norm)
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.nonfinite_streak = 0
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

        self.optimizer.zero_grad(set_to_none=True)

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
        return metrics[self.best_key]  # value the best.pt selection compares on (val/loss or val/viou_mean)

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
        # lambda-VAE schedule state. Its buffers are persistent=False so that checkpoints stay
        # loadable by models built before lambda-VAE existed; that also keeps them out of
        # state_dict, so they are saved separately here or a resume would silently restart the ramp.
        lam_state = {k: getattr(self.model, k).detach().cpu().clone()
                     for k in _LAM_BUFFERS if hasattr(self.model, k)}
        if lam_state:
            state["lam_state"] = lam_state
        path = self.ckpt.save(state, is_best=is_best)
        log.info("saved checkpoint -> %s", path)

    def _nonfinite(self, where: str, out, batch, grad_norm=None) -> None:
        """Report a NaN/Inf and drop the step. Aborts after nonfinite_patience in a row.

        A single bad micro-batch is survivable and worth skipping; a run that cannot produce a
        finite step any more is burning GPU for nothing, which is what the lambda-VAE overflow did
        for hours before anyone noticed. The dump is deliberately cheap (reductions only) so the
        guard costs nothing on the happy path.
        """
        self.nonfinite_streak = getattr(self, "nonfinite_streak", 0) + 1
        self.nonfinite_total = getattr(self, "nonfinite_total", 0) + 1
        if self.nonfinite_streak <= 3 or self.nonfinite_streak % 100 == 0:
            parts = [f"non-finite {where} at epoch {self.epoch} step {self.global_step} "
                     f"(streak {self.nonfinite_streak}, total {self.nonfinite_total})"]
            if grad_norm is not None:
                parts.append(f"grad_norm={float(grad_norm)}")
            parts.append("metrics=" + ", ".join(f"{k}={v:.4g}" for k, v in out.metrics.items()
                                                if isinstance(v, (int, float))))
            for k, v in batch.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    fin = torch.isfinite(v)
                    parts.append(f"{k}[finite={int(fin.all())} absmax={float(v[fin].abs().max()) if fin.any() else float('nan'):.4g}]")
            bad = [n for n, p in self.model.named_parameters() if not torch.isfinite(p).all()]
            parts.append(f"non-finite params: {len(bad)}" + (f" first={bad[0]}" if bad else ""))
            if grad_norm is not None:
                bg = [n for n, p in self.model.named_parameters()
                      if p.grad is not None and not torch.isfinite(p.grad).all()]
                parts.append(f"non-finite grads: {len(bg)}" + (f" first={bg[0]}" if bg else ""))
            log.warning(" | ".join(parts))
        if self.nonfinite_streak >= self.nonfinite_patience:
            raise RuntimeError(
                f"{self.nonfinite_streak} consecutive non-finite steps at epoch {self.epoch} "
                f"step {self.global_step}; aborting instead of burning GPU. Last good checkpoint is "
                f"in {self.ckpt.dir if hasattr(self.ckpt, 'dir') else 'checkpoints/'}."
            )

    def _resume(self, path: str) -> None:
        ckpt = CheckpointManager.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        lrs = [g["lr"] for g in self.optimizer.param_groups]  # config lr built this run, pre-restore
        self.optimizer.load_state_dict(ckpt["optimizer"])     # keeps Adam moments, but also restores old lr
        if self.scheduler is None:
            # No scheduler → nothing re-drives lr, so the checkpoint's lr would silently win. Keep the
            # config lr instead (lets a resume fine-tune at a new constant lr). With a scheduler this is
            # skipped: the scheduler owns lr and re-sets it every step from its restored state.
            for g, lr in zip(self.optimizer.param_groups, lrs):
                g["lr"] = lr
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        for k, v in (ckpt.get("lam_state") or {}).items():
            if hasattr(self.model, k):
                getattr(self.model, k).copy_(v.to(self.device))
        if ckpt.get("lam_state"):
            log.info("restored lambda-VAE schedule: lam_step=%d", int(self.model.lam_step))
        elif hasattr(self.model, "lam_step") and float(getattr(self.model, "lam_delta", 0.0)) > 1.0:
            log.warning("checkpoint has no lambda schedule state; the ramp restarts from step 0")
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

    train_loader, val_loader = build_dataloaders(cfg.data)
    steps_per_epoch = len(train_loader) // max(1, int(tcfg.grad_accum_steps))
    max_epochs = int(tcfg.max_epochs)
    total_steps = steps_per_epoch * max_epochs if max_epochs > 0 else None
    scheduler = build_scheduler(cfg.get("scheduler"), optimizer, total_steps=total_steps)
    log.info("scheduler total_steps (T_max if missing): %s", total_steps)

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
