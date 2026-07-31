from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


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
) -> EpochMetrics:
    model.train()

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

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = loss_function(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size

        total_correct_predictions += calculate_number_of_correct_predictions(
            logits=logits,
            labels=labels,
        )

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The training DataLoader produced no examples")

    average_loss = total_loss / total_examples

    accuracy = total_correct_predictions / total_examples

    return EpochMetrics(
        loss=average_loss,
        accuracy=accuracy,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
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

        logits = model(images)
        loss = loss_function(logits, labels)

        batch_size = images.shape[0]

        total_loss += loss.item() * batch_size

        total_correct_predictions += calculate_number_of_correct_predictions(
            logits=logits,
            labels=labels,
        )

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError("The validation DataLoader produced no examples")

    average_loss = total_loss / total_examples

    accuracy = total_correct_predictions / total_examples

    return EpochMetrics(
        loss=average_loss,
        accuracy=accuracy,
    )
