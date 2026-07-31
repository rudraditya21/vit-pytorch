from __future__ import annotations

import math
from collections.abc import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    minimum_learning_rate: float,
    base_learning_rate: float,
) -> LambdaLR:
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

    minimum_factor = minimum_learning_rate / base_learning_rate

    cosine_epochs = total_epochs - warmup_epochs

    def learning_rate_multiplier(
        epoch_index: int,
    ) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)

        cosine_epoch_index = epoch_index - warmup_epochs

        cosine_progress = min(
            max(
                cosine_epoch_index / max(cosine_epochs - 1, 1),
                0.0,
            ),
            1.0,
        )

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

        return minimum_factor + (1.0 - minimum_factor) * cosine_factor

    scheduler_function: Callable[[int], float] = learning_rate_multiplier

    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=scheduler_function,
    )
