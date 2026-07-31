from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from training.amp import AutomaticMixedPrecision


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def calculate_number_of_correct_predictions(
    logits: Tensor,
    labels: Tensor,
) -> int:
    predictions = logits.argmax(dim=1)

    return int(predictions.eq(labels).sum().item())


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    mixed_precision: AutomaticMixedPrecision,
    gradient_clip_norm: float | None,
) -> EpochMetrics:
    model.train()

    total_loss = 0.0
    total_correct_predictions = 0
    total_examples = 0

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    for images, labels in data_loader:
        images = images.to(
            device=device,
            non_blocking=True,
        )

        labels = labels.to(
            device=device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with mixed_precision.autocast():
            logits = model(images)
            loss = loss_function(
                logits,
                labels,
            )

        mixed_precision.backward_and_step(
            loss=loss,
            optimizer=optimizer,
            parameters=trainable_parameters,
            gradient_clip_norm=gradient_clip_norm,
        )

        batch_size = images.shape[0]

        total_loss += float(loss.detach().item()) * batch_size

        total_correct_predictions += calculate_number_of_correct_predictions(
            logits=logits.detach(),
            labels=labels,
        )

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The training DataLoader produced no examples")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=(total_correct_predictions / total_examples),
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    mixed_precision: AutomaticMixedPrecision,
) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    total_correct_predictions = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(
            device=device,
            non_blocking=True,
        )

        labels = labels.to(
            device=device,
            non_blocking=True,
        )

        with mixed_precision.autocast():
            logits = model(images)
            loss = loss_function(
                logits,
                labels,
            )

        batch_size = images.shape[0]

        total_loss += float(loss.item()) * batch_size

        total_correct_predictions += calculate_number_of_correct_predictions(
            logits=logits,
            labels=labels,
        )

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The validation DataLoader produced no examples")

    return EpochMetrics(
        loss=total_loss / total_examples,
        accuracy=(total_correct_predictions / total_examples),
    )
