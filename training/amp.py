from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler


class AutomaticMixedPrecision:
    def __init__(
        self,
        device: torch.device,
        enabled: bool,
    ) -> None:
        self.device = device
        self.enabled = enabled and device.type == "cuda"

        self.scaler = GradScaler(
            "cuda",
            enabled=self.enabled,
        )

    def autocast(
        self,
    ) -> AbstractContextManager[Any]:
        if not self.enabled:
            return nullcontext()

        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )

    def backward_and_step(
        self,
        loss: Tensor,
        optimizer: Optimizer,
        parameters: Iterable[Tensor],
        gradient_clip_norm: float | None,
    ) -> None:
        self.scaler.scale(loss).backward()

        if gradient_clip_norm is not None:
            self.scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                parameters=parameters,
                max_norm=gradient_clip_norm,
            )

        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict[str, Any]:
        return self.scaler.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self.scaler.load_state_dict(state_dict)
