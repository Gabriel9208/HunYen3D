from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class StepOutput:
    """Result of a single forward step.

    Args:
        loss: scalar tensor to backprop.
        metrics: extra scalars to log (e.g. {"kl": ..., "recon": ...}).
    """

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    """Decouples the training loop from model specifics.

    A future VAETask would run encode/decode and return recon + KL here,
    without the Trainer needing any changes.
    """

    def step(self, model: nn.Module, batch) -> StepOutput: ...


class BaseTask(ABC):
    @abstractmethod
    def step(self, model: nn.Module, batch) -> StepOutput: ...


class DummyTask(BaseTask):
    """Trivial regression task so the scaffold runs end-to-end without real data."""

    def __init__(self) -> None:
        self.criterion = nn.MSELoss()

    def step(self, model: nn.Module, batch) -> StepOutput:
        x, y = batch["input"], batch["target"]
        pred = model(x)
        loss = self.criterion(pred, y)
        return StepOutput(loss=loss, metrics={"mse": loss.item()})


class VAETask(BaseTask):
    """Shape VAE training task.

    Wires the encode side: `batch["query"]`/`batch["data"]` come from
    `Preprocessor` (Fourier-embedded surface points). `model.encode` returns the
    sampled latent `z` (B, num_latents, latent_dim) and per-sample `kl` (B,).

    The reconstruction/decode + final loss is intentionally LEFT TO THE USER in
    `compute_loss` (see the NotImplementedError). `kl_weight` is the `r` in
    `recon + r * KL` and is config-driven.
    """

    def __init__(self, kl_weight: float = 1.0e-3) -> None:
        self.kl_weight = kl_weight

    def step(self, model: nn.Module, batch) -> StepOutput:
        z, kl = model.encode(batch["query"], batch["data"])
        
        return self.compute_loss(model, batch, z, kl)

    def compute_loss(self, model: nn.Module, batch, z, kl) -> StepOutput:
        pred_sdf = model.decode(z, batch["query_points"])
        recon_loss = F.mse_loss(pred_sdf, batch["gt_sdf"])

        loss = recon_loss + self.kl_weight * kl.mean()
        
        return StepOutput(
            loss=loss,
            metrics={
                "recon": recon_loss.item(),
                "kl": kl.mean().item()
            }
        )
