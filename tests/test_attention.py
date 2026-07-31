import pytest
import torch

from vit.attention import MultiHeadSelfAttention


def test_attention_output_shape() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
    )

    tokens = torch.randn(8, 65, 192)

    output = attention(tokens)

    assert output.shape == (8, 65, 192)


def test_attention_weights_shape() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
        attention_dropout=0.0,
    )

    tokens = torch.randn(8, 65, 192)

    output, weights = attention(
        tokens,
        return_attention=True,
    )

    assert output.shape == (8, 65, 192)
    assert weights.shape == (8, 3, 65, 65)


def test_attention_weights_sum_to_one() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
        attention_dropout=0.0,
    )

    attention.eval()

    tokens = torch.randn(4, 65, 192)

    _, weights = attention(
        tokens,
        return_attention=True,
    )

    weight_sums = weights.sum(dim=-1)
    expected = torch.ones_like(weight_sums)

    torch.testing.assert_close(
        weight_sums,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def test_attention_preserves_sequence_length() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=128,
        number_of_heads=8,
    )

    for sequence_length in (1, 16, 65, 197):
        tokens = torch.randn(
            2,
            sequence_length,
            128,
        )

        output = attention(tokens)

        assert output.shape == (
            2,
            sequence_length,
            128,
        )


def test_attention_receives_gradients() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
    )

    tokens = torch.randn(
        2,
        65,
        192,
        requires_grad=True,
    )

    output = attention(tokens)

    loss = output.square().mean()
    loss.backward()

    assert tokens.grad is not None
    assert attention.qkv_projection.weight.grad is not None
    assert attention.output_projection.weight.grad is not None


def test_attention_rejects_incompatible_head_count() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(
            embedding_dim=192,
            number_of_heads=5,
        )


def test_attention_rejects_incorrect_embedding_dimension() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
    )

    tokens = torch.randn(8, 65, 128)

    with pytest.raises(ValueError):
        attention(tokens)


def test_attention_rejects_invalid_tensor_rank() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
    )

    tokens = torch.randn(8, 192)

    with pytest.raises(ValueError):
        attention(tokens)


def test_attention_is_deterministic_without_dropout() -> None:
    attention = MultiHeadSelfAttention(
        embedding_dim=192,
        number_of_heads=3,
        attention_dropout=0.0,
        projection_dropout=0.0,
    )

    attention.eval()

    tokens = torch.randn(2, 65, 192)

    first_output = attention(tokens)
    second_output = attention(tokens)

    torch.testing.assert_close(
        first_output,
        second_output,
    )
