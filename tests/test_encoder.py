import pytest
import torch
from torch import nn

from vit.encoder import TransformerEncoderBlock, TransformerEncoder


def test_encoder_block_output_shape() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 192)

    output = block(tokens)

    assert output.shape == (8, 65, 192)


def test_encoder_block_returns_attention_weights() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
        attention_dropout=0.0,
    )

    tokens = torch.randn(8, 65, 192)

    output, attention_weights = block(
        tokens,
        return_attention=True,
    )

    assert output.shape == (8, 65, 192)
    assert attention_weights.shape == (8, 3, 65, 65)


def test_encoder_block_attention_weights_sum_to_one() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
        attention_dropout=0.0,
    )

    block.eval()

    tokens = torch.randn(4, 65, 192)

    _, attention_weights = block(
        tokens,
        return_attention=True,
    )

    sums = attention_weights.sum(dim=-1)

    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        rtol=1e-5,
        atol=1e-6,
    )


def test_encoder_block_preserves_sequence_length() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=128,
        number_of_heads=8,
        mlp_hidden_dim=512,
    )

    for sequence_length in (1, 16, 65, 197):
        tokens = torch.randn(
            2,
            sequence_length,
            128,
        )

        output = block(tokens)

        assert output.shape == (
            2,
            sequence_length,
            128,
        )


def test_encoder_block_receives_gradients() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(
        2,
        65,
        192,
        requires_grad=True,
    )

    output = block(tokens)
    loss = output.square().mean()

    loss.backward()

    assert tokens.grad is not None

    assert block.attention.qkv_projection.weight.grad is not None

    assert block.attention.output_projection.weight.grad is not None

    assert block.mlp.input_projection.weight.grad is not None

    assert block.mlp.output_projection.weight.grad is not None

    assert block.attention_norm.weight.grad is not None
    assert block.mlp_norm.weight.grad is not None


def test_encoder_block_rejects_wrong_embedding_dimension() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 128)

    with pytest.raises(ValueError):
        block(tokens)


def test_encoder_block_rejects_wrong_tensor_rank() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(8, 192)

    with pytest.raises(ValueError):
        block(tokens)


def test_encoder_block_has_independent_layer_norms() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    assert isinstance(block.attention_norm, nn.LayerNorm)
    assert isinstance(block.mlp_norm, nn.LayerNorm)

    assert block.attention_norm is not block.mlp_norm
    assert block.attention_norm.weight is not block.mlp_norm.weight


def test_encoder_block_is_deterministic_in_eval_mode() -> None:
    block = TransformerEncoderBlock(
        embedding_dim=192,
        number_of_heads=3,
        mlp_hidden_dim=768,
        attention_dropout=0.5,
        projection_dropout=0.5,
        mlp_dropout=0.5,
    )

    block.eval()

    tokens = torch.randn(2, 65, 192)

    first_output = block(tokens)
    second_output = block(tokens)

    torch.testing.assert_close(
        first_output,
        second_output,
    )


def test_transformer_encoder_output_shape() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 192)

    output = encoder(tokens)

    assert output.shape == (8, 65, 192)


def test_transformer_encoder_contains_requested_layers() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    assert len(encoder.blocks) == 6

    assert all(isinstance(block, TransformerEncoderBlock) for block in encoder.blocks)


def test_transformer_encoder_blocks_are_independent() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    first_weight = encoder.blocks[0].attention.qkv_projection.weight

    second_weight = encoder.blocks[1].attention.qkv_projection.weight

    assert first_weight is not second_weight


def test_transformer_encoder_returns_all_attention_maps() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
        attention_dropout=0.0,
    )

    encoder.eval()

    tokens = torch.randn(8, 65, 192)

    output, attention_maps = encoder(
        tokens,
        return_attention=True,
    )

    assert output.shape == (8, 65, 192)

    assert attention_maps.shape == (
        6,
        8,
        3,
        65,
        65,
    )


def test_all_encoder_attention_maps_sum_to_one() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
        attention_dropout=0.0,
    )

    encoder.eval()

    tokens = torch.randn(2, 65, 192)

    _, attention_maps = encoder(
        tokens,
        return_attention=True,
    )

    sums = attention_maps.sum(dim=-1)

    torch.testing.assert_close(
        sums,
        torch.ones_like(sums),
        rtol=1e-5,
        atol=1e-6,
    )


def test_transformer_encoder_registers_all_parameters() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    parameter_names = {name for name, _ in encoder.named_parameters()}

    for layer_index in range(6):
        expected_name = f"blocks.{layer_index}.attention.qkv_projection.weight"

        assert expected_name in parameter_names


def test_transformer_encoder_receives_gradients() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(
        2,
        65,
        192,
        requires_grad=True,
    )

    output = encoder(tokens)
    loss = output.square().mean()

    loss.backward()

    assert tokens.grad is not None
    assert encoder.final_norm.weight.grad is not None

    for block in encoder.blocks:
        assert block.attention.qkv_projection.weight.grad is not None

        assert block.mlp.input_projection.weight.grad is not None


def test_transformer_encoder_final_norm_is_applied() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=2,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    encoder.eval()

    tokens = torch.randn(4, 65, 192)
    output = encoder(tokens)

    means = output.mean(dim=-1)
    variances = output.var(
        dim=-1,
        unbiased=False,
    )

    torch.testing.assert_close(
        means,
        torch.zeros_like(means),
        atol=1e-5,
        rtol=1e-5,
    )

    torch.testing.assert_close(
        variances,
        torch.ones_like(variances),
        atol=1e-4,
        rtol=1e-4,
    )


def test_transformer_encoder_rejects_invalid_layer_count() -> None:
    with pytest.raises(ValueError):
        TransformerEncoder(
            embedding_dim=192,
            number_of_layers=0,
            number_of_heads=3,
            mlp_hidden_dim=768,
        )


def test_transformer_encoder_rejects_wrong_input_dimension() -> None:
    encoder = TransformerEncoder(
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
    )

    tokens = torch.randn(8, 65, 128)

    with pytest.raises(ValueError):
        encoder(tokens)
