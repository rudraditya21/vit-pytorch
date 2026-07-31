from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TrainingConfig:
    # Reproducibility
    random_seed: int = 42

    # Dataset
    dataset_directory: Path = Path("data")
    image_size: int = 32
    in_channels: int = 3
    number_of_classes: int = 10

    # Data loading
    batch_size: int = 128
    number_of_workers: int = 4
    pin_memory: bool = True

    # Vision Transformer
    patch_size: int = 4
    embedding_dim: int = 192
    number_of_layers: int = 6
    number_of_heads: int = 3
    mlp_hidden_dim: int = 768

    # Regularization
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    projection_dropout: float = 0.1
    mlp_dropout: float = 0.1

    # Optimization
    number_of_epochs: int = 100
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 1e-6
    weight_decay: float = 0.05
    gradient_clip_norm: float | None = 1.0

    # Learning-rate scheduling
    warmup_epochs: int = 5

    # Automatic mixed precision
    use_mixed_precision: bool = True

    # Checkpoints
    checkpoint_directory: Path = Path("checkpoints")
    checkpoint_interval: int = 10
    resume_checkpoint: Path | None = None

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")

        if torch.backends.mps.is_available():
            return torch.device("mps")

        return torch.device("cpu")

    def validate(self) -> None:
        if self.number_of_epochs <= 0:
            raise ValueError("number_of_epochs must be greater than zero")

        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")

        if self.minimum_learning_rate < 0.0:
            raise ValueError("minimum_learning_rate cannot be negative")

        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")

        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs cannot be negative")

        if self.warmup_epochs >= self.number_of_epochs:
            raise ValueError("warmup_epochs must be smaller than number_of_epochs")

        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be greater than zero")

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be greater than zero")
