from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from training.amp import AutomaticMixedPrecision


@dataclass(frozen=True)
class CheckpointState:
    completed_epoch: int
    best_validation_accuracy: float


class CheckpointManager:
    def __init__(
        self,
        directory: Path,
        checkpoint_interval: int,
    ) -> None:
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than zero")

        self.directory = directory
        self.checkpoint_interval = checkpoint_interval

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.pt"

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def save(
        self,
        *,
        completed_epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        mixed_precision: AutomaticMixedPrecision,
        best_validation_accuracy: float,
        is_best: bool,
    ) -> None:
        checkpoint = self._create_checkpoint(
            completed_epoch=completed_epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mixed_precision=mixed_precision,
            best_validation_accuracy=(best_validation_accuracy),
        )

        self._atomic_save(
            checkpoint=checkpoint,
            destination=self.latest_path,
        )

        if is_best:
            self._atomic_save(
                checkpoint=checkpoint,
                destination=self.best_path,
            )

        if completed_epoch % self.checkpoint_interval == 0:
            epoch_path = self.directory / (f"epoch_{completed_epoch:03d}.pt")

            self._atomic_save(
                checkpoint=checkpoint,
                destination=epoch_path,
            )

    def load(
        self,
        *,
        checkpoint_path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        mixed_precision: AutomaticMixedPrecision,
        device: torch.device,
    ) -> CheckpointState:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        self._validate_checkpoint(checkpoint)

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        mixed_precision.load_state_dict(checkpoint["amp_scaler_state_dict"])

        random.setstate(checkpoint["python_random_state"])

        torch.set_rng_state(checkpoint["torch_random_state"])

        cuda_random_state = checkpoint.get("cuda_random_state")

        if cuda_random_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_random_state)

        return CheckpointState(
            completed_epoch=int(checkpoint["completed_epoch"]),
            best_validation_accuracy=float(checkpoint["best_validation_accuracy"]),
        )

    def _create_checkpoint(
        self,
        *,
        completed_epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Any,
        mixed_precision: AutomaticMixedPrecision,
        best_validation_accuracy: float,
    ) -> dict[str, Any]:
        cuda_random_state = None

        if torch.cuda.is_available():
            cuda_random_state = torch.cuda.get_rng_state_all()

        return {
            "checkpoint_version": 1,
            "completed_epoch": completed_epoch,
            "best_validation_accuracy": (best_validation_accuracy),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (optimizer.state_dict()),
            "scheduler_state_dict": (scheduler.state_dict()),
            "amp_scaler_state_dict": (mixed_precision.state_dict()),
            "python_random_state": (random.getstate()),
            "torch_random_state": (torch.get_rng_state()),
            "cuda_random_state": (cuda_random_state),
        }

    @staticmethod
    def _atomic_save(
        *,
        checkpoint: dict[str, Any],
        destination: Path,
    ) -> None:
        temporary_path = destination.with_suffix(destination.suffix + ".tmp")

        torch.save(
            checkpoint,
            temporary_path,
        )

        temporary_path.replace(destination)

    @staticmethod
    def _validate_checkpoint(
        checkpoint: Any,
    ) -> None:
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Checkpoint must contain a dictionary")

        required_keys = {
            "completed_epoch",
            "best_validation_accuracy",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "amp_scaler_state_dict",
            "python_random_state",
            "torch_random_state",
        }

        missing_keys = required_keys.difference(checkpoint)

        if missing_keys:
            missing = ", ".join(sorted(missing_keys))

            raise RuntimeError(f"Checkpoint is missing keys: {missing}")
