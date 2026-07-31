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

    # Data Loading
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
    number_of_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 0.05

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
