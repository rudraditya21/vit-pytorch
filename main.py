from __future__ import annotations

import random

import torch
from torch import nn
from torch.optim import AdamW

from training import (
    AutomaticMixedPrecision,
    CheckpointManager,
    TrainingConfig,
    create_data_loaders,
    create_warmup_cosine_scheduler,
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
        number_of_classes=(config.number_of_classes),
        embedding_dim=config.embedding_dim,
        number_of_layers=config.number_of_layers,
        number_of_heads=config.number_of_heads,
        mlp_hidden_dim=config.mlp_hidden_dim,
        embedding_dropout=(config.embedding_dropout),
        attention_dropout=(config.attention_dropout),
        projection_dropout=(config.projection_dropout),
        mlp_dropout=config.mlp_dropout,
    )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def get_learning_rate(
    optimizer: AdamW,
) -> float:
    return float(optimizer.param_groups[0]["lr"])


def print_training_header(
    config: TrainingConfig,
    model: nn.Module,
    mixed_precision: AutomaticMixedPrecision,
) -> None:
    print("=" * 80)
    print("Vision Transformer Training")
    print("=" * 80)
    print(f"Device               : {config.device}")
    print(f"Mixed precision      : {mixed_precision.enabled}")
    print(f"Trainable parameters : {count_trainable_parameters(model):,}")
    print(f"Epochs               : {config.number_of_epochs}")
    print(f"Warmup epochs        : {config.warmup_epochs}")
    print(f"Batch size           : {config.batch_size}")
    print(f"Base learning rate   : {config.learning_rate:.8f}")
    print(f"Minimum learning rate: {config.minimum_learning_rate:.8f}")
    print("=" * 80)
    print()


def main() -> None:
    config = TrainingConfig()
    config.validate()

    set_random_seed(config.random_seed)

    data_loaders = create_data_loaders(config)

    model = create_model(config)
    model = model.to(config.device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=config.number_of_epochs,
        warmup_epochs=config.warmup_epochs,
        minimum_learning_rate=(config.minimum_learning_rate),
        base_learning_rate=config.learning_rate,
    )

    mixed_precision = AutomaticMixedPrecision(
        device=config.device,
        enabled=config.use_mixed_precision,
    )

    checkpoint_manager = CheckpointManager(
        directory=config.checkpoint_directory,
        checkpoint_interval=(config.checkpoint_interval),
    )

    starting_epoch = 1
    best_validation_accuracy = 0.0

    if config.resume_checkpoint is not None:
        checkpoint_state = checkpoint_manager.load(
            checkpoint_path=(config.resume_checkpoint),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mixed_precision=mixed_precision,
            device=config.device,
        )

        starting_epoch = checkpoint_state.completed_epoch + 1

        best_validation_accuracy = checkpoint_state.best_validation_accuracy

        print(f"Resumed from checkpoint: {config.resume_checkpoint}")
        print(f"Continuing from epoch: {starting_epoch}")
        print()

    print_training_header(
        config=config,
        model=model,
        mixed_precision=mixed_precision,
    )

    if starting_epoch > config.number_of_epochs:
        print("Training is already complete for the configured number of epochs.")
        return

    for epoch in range(
        starting_epoch,
        config.number_of_epochs + 1,
    ):
        current_learning_rate = get_learning_rate(optimizer)

        training_metrics = train_one_epoch(
            model=model,
            data_loader=data_loaders.train,
            loss_function=loss_function,
            optimizer=optimizer,
            device=config.device,
            mixed_precision=mixed_precision,
            gradient_clip_norm=(config.gradient_clip_norm),
        )

        validation_metrics = validate_one_epoch(
            model=model,
            data_loader=(data_loaders.validation),
            loss_function=loss_function,
            device=config.device,
            mixed_precision=mixed_precision,
        )

        is_best = validation_metrics.accuracy > best_validation_accuracy

        if is_best:
            best_validation_accuracy = validation_metrics.accuracy

        # Advance the scheduler for the next epoch.
        scheduler.step()

        checkpoint_manager.save(
            completed_epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            mixed_precision=mixed_precision,
            best_validation_accuracy=(best_validation_accuracy),
            is_best=is_best,
        )

        print(
            f"Epoch "
            f"[{epoch:03d}/"
            f"{config.number_of_epochs:03d}] "
            f"| LR: {current_learning_rate:.8f} "
            f"| Train Loss: "
            f"{training_metrics.loss:.4f} "
            f"| Train Accuracy: "
            f"{training_metrics.accuracy * 100:.2f}% "
            f"| Validation Loss: "
            f"{validation_metrics.loss:.4f} "
            f"| Validation Accuracy: "
            f"{validation_metrics.accuracy * 100:.2f}% "
            f"| Best: "
            f"{best_validation_accuracy * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
