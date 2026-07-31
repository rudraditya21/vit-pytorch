import pytest
import torch
from torch import nn

from vit.model import VisionTransformer


def create_model() -> VisionTransformer:
    return VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        number_of_classes=10,
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        projection_dropout=0.0,
        mlp_dropout=0.0,
    )


def test_model_output_shape() -> None:
    model = create_model()

    images = torch.randn(8, 3, 32, 32)

    logits = model(images)

    assert logits.shape == (8, 10)


def test_model_returns_attention_maps() -> None:
    model = create_model()
    model.eval()

    images = torch.randn(8, 3, 32, 32)

    logits, attention_maps = model(
        images,
        return_attention=True,
    )

    assert logits.shape == (8, 10)

    assert attention_maps.shape == (
        6,
        8,
        3,
        65,
        65,
    )


def test_model_attention_weights_sum_to_one() -> None:
    model = create_model()
    model.eval()

    images = torch.randn(2, 3, 32, 32)

    _, attention_maps = model(
        images,
        return_attention=True,
    )

    sums = attention_maps.sum(dim=-1)

    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        rtol=1e-5,
        atol=1e-6,
    )


def test_model_receives_gradients() -> None:
    model = create_model()

    images = torch.randn(
        2,
        3,
        32,
        32,
        requires_grad=True,
    )

    labels = torch.tensor([2, 7])

    logits = model(images)

    loss = nn.CrossEntropyLoss()(
        logits,
        labels,
    )

    loss.backward()

    assert images.grad is not None

    assert model.classification_head.weight.grad is not None

    assert model.embeddings.patch_embedding.projection.weight.grad is not None

    assert model.embeddings.class_token.grad is not None

    assert model.embeddings.position_embeddings.grad is not None

    for block in model.encoder.blocks:
        assert block.attention.qkv_projection.weight.grad is not None

        assert block.mlp.input_projection.weight.grad is not None


def test_model_classification_head_is_zero_initialized() -> None:
    model = create_model()

    torch.testing.assert_close(
        model.classification_head.weight,
        torch.zeros_like(model.classification_head.weight),
    )

    torch.testing.assert_close(
        model.classification_head.bias,
        torch.zeros_like(model.classification_head.bias),
    )


def test_initial_logits_are_zero() -> None:
    model = create_model()
    model.eval()

    images = torch.randn(4, 3, 32, 32)

    logits = model(images)

    torch.testing.assert_close(
        logits,
        torch.zeros_like(logits),
    )


def test_model_rejects_invalid_image_resolution() -> None:
    model = create_model()

    images = torch.randn(8, 3, 64, 64)

    with pytest.raises(ValueError):
        model(images)


def test_model_rejects_invalid_channel_count() -> None:
    model = create_model()

    images = torch.randn(8, 1, 32, 32)

    with pytest.raises(ValueError):
        model(images)


def test_model_rejects_invalid_class_count() -> None:
    with pytest.raises(ValueError):
        VisionTransformer(
            image_size=32,
            patch_size=4,
            in_channels=3,
            number_of_classes=0,
            embedding_dim=192,
            number_of_layers=6,
            number_of_heads=3,
            mlp_hidden_dim=768,
        )


def test_model_is_deterministic_in_eval_mode() -> None:
    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        number_of_classes=10,
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
        embedding_dropout=0.5,
        attention_dropout=0.5,
        projection_dropout=0.5,
        mlp_dropout=0.5,
    )

    model.eval()

    images = torch.randn(2, 3, 32, 32)

    first_logits = model(images)
    second_logits = model(images)

    torch.testing.assert_close(
        first_logits,
        second_logits,
    )
