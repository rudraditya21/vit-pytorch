from __future__ import annotations

import math

from torch.optim import Optimizer


class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        warmup_epochs: int,
        minimum_learning_rate: float,
        base_learning_rate: float,
    ) -> None:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")

        if warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")

        if warmup_epochs >= total_epochs:
            raise ValueError("warmup_epochs must be smaller than total_epochs")

        if base_learning_rate <= 0.0:
            raise ValueError("base_learning_rate must be greater than zero")

        if minimum_learning_rate < 0.0:
            raise ValueError("minimum_learning_rate cannot be negative")

        if minimum_learning_rate > base_learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed base_learning_rate")

        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.minimum_learning_rate = minimum_learning_rate
        self.base_learning_rate = base_learning_rate
        self.minimum_factor = minimum_learning_rate / base_learning_rate
        self.cosine_epochs = total_epochs - warmup_epochs
        self.base_lrs = [
            base_learning_rate for _ in self.optimizer.param_groups
        ]
        self.last_epoch = 0
        self._last_lr: list[float] = []

        self._set_learning_rate_for_epoch(epoch_index=0)

    def step(self) -> None:
        self.last_epoch += 1
        self._set_learning_rate_for_epoch(epoch_index=self.last_epoch)

    def state_dict(self) -> dict[str, int | float | list[float]]:
        return {
            "total_epochs": self.total_epochs,
            "warmup_epochs": self.warmup_epochs,
            "minimum_learning_rate": self.minimum_learning_rate,
            "base_learning_rate": self.base_learning_rate,
            "last_epoch": self.last_epoch,
            "_last_lr": list(self._last_lr),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, int | float | list[float]],
    ) -> None:
        self.last_epoch = int(state_dict["last_epoch"])

        self._last_lr = [float(lr) for lr in state_dict["_last_lr"]]

        if len(self._last_lr) != len(self.optimizer.param_groups):
            raise RuntimeError("Scheduler state is incompatible with optimizer")

        for parameter_group, learning_rate in zip(
            self.optimizer.param_groups,
            self._last_lr,
        ):
            parameter_group["lr"] = learning_rate

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def _set_learning_rate_for_epoch(
        self,
        epoch_index: int,
    ) -> None:
        multiplier = self._learning_rate_multiplier(epoch_index=epoch_index)
        learning_rate = self.base_learning_rate * multiplier
        self._last_lr = [learning_rate for _ in self.optimizer.param_groups]

        for parameter_group, parameter_group_learning_rate in zip(
            self.optimizer.param_groups,
            self._last_lr,
        ):
            parameter_group["lr"] = parameter_group_learning_rate

    def _learning_rate_multiplier(
        self,
        epoch_index: int,
    ) -> float:
        if self.warmup_epochs > 0 and epoch_index < self.warmup_epochs:
            return float(epoch_index + 1) / float(self.warmup_epochs)

        cosine_epoch_index = epoch_index - self.warmup_epochs

        cosine_progress = min(
            max(
                cosine_epoch_index / max(self.cosine_epochs - 1, 1),
                0.0,
            ),
            1.0,
        )

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

        return self.minimum_factor + (1.0 - self.minimum_factor) * cosine_factor


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
    base_learning_rate: float,
) -> WarmupCosineScheduler:
    return WarmupCosineScheduler(
        optimizer=optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        minimum_learning_rate=minimum_learning_rate,
        base_learning_rate=base_learning_rate,
    )
