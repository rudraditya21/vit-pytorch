import pytest
import torch

from vit.mlp import MLP


def test_mlp_output_shape() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
        dropout=0.0,
    )

    tokens = torch.randn(8, 65, 192)

    output = mlp(tokens)

    assert output.shape == (8, 65, 192)


def test_mlp_expands_to_hidden_dimension() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 192)

    hidden = mlp.input_projection(tokens)

    assert hidden.shape == (8, 65, 768)


def test_mlp_preserves_sequence_length() -> None:
    mlp = MLP(
        embedding_dim=128,
        hidden_dim=512,
    )

    for sequence_length in (1, 16, 65, 197):
        tokens = torch.randn(
            2,
            sequence_length,
            128,
        )

        output = mlp(tokens)

        assert output.shape == (
            2,
            sequence_length,
            128,
        )


def test_mlp_receives_gradients() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
    )

    tokens = torch.randn(
        2,
        65,
        192,
        requires_grad=True,
    )

    output = mlp(tokens)
    loss = output.square().mean()

    loss.backward()

    assert tokens.grad is not None
    assert mlp.input_projection.weight.grad is not None
    assert mlp.output_projection.weight.grad is not None


def test_mlp_rejects_incorrect_embedding_dimension() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 128)

    with pytest.raises(ValueError):
        mlp(tokens)


def test_mlp_rejects_invalid_tensor_rank() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
    )

    tokens = torch.randn(8, 192)

    with pytest.raises(ValueError):
        mlp(tokens)


def test_mlp_rejects_invalid_dropout() -> None:
    with pytest.raises(ValueError):
        MLP(
            embedding_dim=192,
            hidden_dim=768,
            dropout=1.0,
        )


def test_mlp_is_deterministic_without_dropout() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
        dropout=0.0,
    )

    mlp.eval()

    tokens = torch.randn(2, 65, 192)

    first_output = mlp(tokens)
    second_output = mlp(tokens)

    torch.testing.assert_close(
        first_output,
        second_output,
    )


def test_dropout_is_disabled_in_evaluation_mode() -> None:
    mlp = MLP(
        embedding_dim=192,
        hidden_dim=768,
        dropout=0.5,
    )

    mlp.eval()

    tokens = torch.randn(2, 65, 192)

    first_output = mlp(tokens)
    second_output = mlp(tokens)

    torch.testing.assert_close(
        first_output,
        second_output,
    )
