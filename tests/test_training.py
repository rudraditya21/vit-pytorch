import pytest
import torch
from torch import nn
from torch.optim import AdamW

from training.config import TrainingConfig
from training.scheduler import create_warmup_cosine_scheduler
from vit.attention import MultiHeadSelfAttention


def test_scheduler_sets_first_warmup_learning_rate_immediately() -> None:
    model = nn.Linear(4, 2)
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=5,
        minimum_learning_rate=1e-6,
        base_learning_rate=3e-4,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4 / 5.0)

    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.0 * 3e-4 / 5.0)


def test_scheduler_restores_next_epoch_learning_rate_from_state() -> None:
    model = nn.Linear(4, 2)
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
    )

    scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_epochs=10,
        warmup_epochs=5,
        minimum_learning_rate=1e-6,
        base_learning_rate=3e-4,
    )

    scheduler.step()
    scheduler.step()

    saved_state = scheduler.state_dict()

    restored_optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
    )

    restored_scheduler = create_warmup_cosine_scheduler(
        optimizer=restored_optimizer,
        total_epochs=10,
        warmup_epochs=5,
        minimum_learning_rate=1e-6,
        base_learning_rate=3e-4,
    )

    restored_scheduler.load_state_dict(saved_state)

    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )


def test_training_config_rejects_non_rgb_cifar10() -> None:
    config = TrainingConfig(in_channels=1)

    with pytest.raises(ValueError, match="in_channels"):
        config.validate()


def test_training_config_rejects_non_cifar10_class_count() -> None:
    config = TrainingConfig(number_of_classes=100)

    with pytest.raises(ValueError, match="number_of_classes"):
        config.validate()


def test_returned_attention_maps_remain_probabilities_with_dropout() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
        attention_dropout=0.5,
    )
    attention.train()

    tokens = torch.randn(2, 65, 192)

    _, attention_maps = attention(
        tokens,
        return_attention=True,
    )

    sums = attention_maps.sum(dim=-1)

    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        atol=1e-6,
        rtol=1e-5,
    )
