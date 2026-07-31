from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from training.config import TrainingConfig


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader


def create_data_loaders(config: TrainingConfig) -> DataLoaders:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(
                size=config.image_size,
                padding=4,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    training_dataset = datasets.CIFAR10(
        root=config.dataset_directory,
        train=True,
        transform=train_transform,
        download=True,
    )

    validation_dataset = datasets.CIFAR10(
        root=config.dataset_directory,
        train=False,
        transform=validation_transform,
        download=True,
    )

    persistent_workers = config.number_of_workers > 0

    training_loader = DataLoader(
        dataset=training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.number_of_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.number_of_workers,
        pin_memory=config.pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return DataLoaders(
        train=training_loader,
        validation=validation_loader,
    )
