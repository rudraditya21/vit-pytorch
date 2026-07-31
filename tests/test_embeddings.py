import pytest
import torch

from vit.embeddings import PatchEmbedding, ViTEmbeddings


def test_patch_embedding_output_shape() -> None:
    layer = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
    )

    images = torch.randn(8, 3, 32, 32)
    output = layer(images)

    assert output.shape == (8, 64, 192)


def test_patch_embedding_number_of_patches() -> None:
    layer = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
    )

    assert layer.grid_size == 8
    assert layer.number_of_patches == 64


def test_patch_embedding_rejects_invalid_image_size() -> None:
    with pytest.raises(ValueError):
        PatchEmbedding(
            image_size=30,
            patch_size=4,
            in_channels=3,
            embedding_dim=192,
        )


def test_patch_embedding_rejects_incorrect_channels() -> None:
    layer = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
    )

    images = torch.randn(8, 1, 32, 32)

    with pytest.raises(ValueError):
        layer(images)


def test_patch_embedding_rejects_incorrect_resolution() -> None:
    layer = PatchEmbedding(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
    )

    images = torch.randn(8, 3, 64, 64)

    with pytest.raises(ValueError):
        layer(images)


def test_vit_embeddings_output_shape() -> None:
    layer = ViTEmbeddings(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
        dropout=0.0,
    )

    images = torch.randn(8, 3, 32, 32)
    output = layer(images)

    assert output.shape == (8, 65, 192)


def test_vit_embeddings_sequence_length() -> None:
    layer = ViTEmbeddings(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
    )

    assert layer.sequence_length == 65


def test_class_token_is_first_token() -> None:
    layer = ViTEmbeddings(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
        dropout=0.0,
    )

    images = torch.randn(2, 3, 32, 32)
    output = layer(images)

    expected_class_token = (
        layer.class_token + layer.position_embeddings[:, :1]
    ).expand(2, -1, -1)

    torch.testing.assert_close(
        output[:, :1],
        expected_class_token,
    )


def test_learnable_embeddings_receive_gradients() -> None:
    layer = ViTEmbeddings(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embedding_dim=192,
        dropout=0.0,
    )

    images = torch.randn(2, 3, 32, 32)

    output = layer(images)
    loss = output.square().mean()
    loss.backward()

    assert layer.class_token.grad is not None
    assert layer.position_embeddings.grad is not None
    assert layer.patch_embedding.projection.weight.grad is not None


def test_invalid_dropout_is_rejected() -> None:
    with pytest.raises(ValueError):
        ViTEmbeddings(
            image_size=32,
            patch_size=4,
            in_channels=3,
            embedding_dim=192,
            dropout=1.0,
        )
