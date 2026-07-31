from __future__ import annotations

import random

import torch
from torch import nn
from torch.optim import AdamW

from training import (
    TrainingConfig,
    create_data_loaders,
    train_one_epoch,
    validate_one_epoch,
)
from vit import VisionTransformer


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_model(
    config: TrainingConfig,
) -> VisionTransformer:
    return VisionTransformer(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        number_of_classes=config.number_of_classes,
        embedding_dim=config.embedding_dim,
        number_of_layers=config.number_of_layers,
        number_of_heads=config.number_of_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
        embedding_dropout=config.embedding_dropout,
        attention_dropout=config.attention_dropout,
        projection_dropout=config.projection_dropout,
        mlp_dropout=config.mlp_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def main() -> None:
    config = TrainingConfig()

    set_random_seed(config.random_seed)

    print("=" * 72)
    print("Vision Transformer Training")
    print("=" * 72)
    print(f"Device: {config.device}")
    print(f"Epochs: {config.number_of_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print()

    data_loaders = create_data_loaders(config)

    model = create_model(config)
    model = model.to(config.device)

    print(f"Trainable parameters: {count_parameters(model):,}")
    print()

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    for epoch in range(
        1,
        config.number_of_epochs + 1,
    ):
        training_metrics = train_one_epoch(
            model=model,
            data_loader=data_loaders.train,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            data_loader=data_loaders.validation,
            loss_function=loss_function,
            device=config.device,
        )

        print(
            f"Epoch [{epoch:03d}/{config.number_of_epochs:03d}] "
            f"| "
            f"Train Loss: {training_metrics.loss:.4f} "
            f"| "
            f"Train Accuracy: "
            f"{training_metrics.accuracy * 100:.2f}% "
            f"| "
            f"Validation Loss: "
            f"{validation_metrics.loss:.4f} "
            f"| "
            f"Validation Accuracy: "
            f"{validation_metrics.accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
