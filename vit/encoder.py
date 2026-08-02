from __future__ import annotations

import torch
from torch import Tensor, nn

from vit.attention import MultiHeadSelfAttention
from vit.mlp import MLP


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        number_of_heads: int,
        mlp_hidden_dim: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qkv_bias: bool = True,
        layer_norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be greater than zero")

        if layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be greater than zero")

        self.embedding_dim = embedding_dim

        self.attention_norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=layer_norm_epsilon,
        )

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            number_of_heads=number_of_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            qkv_bias=qkv_bias,
        )

        self.mlp_norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=layer_norm_epsilon,
        )

        self.mlp = MLP(
            embedding_dim=embedding_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
        )

    def forward(
        self,
        tokens: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate_input(tokens)

        if return_attention:
            attention_output, attention_weights = self.attention(
                self.attention_norm(tokens),
                return_attention=True,
            )
        else:
            attention_output = self.attention(self.attention_norm(tokens))
        tokens = tokens + attention_output

        mlp_output = self.mlp(self.mlp_norm(tokens))
        tokens = tokens + mlp_output

        if return_attention:
            return tokens, attention_weights

        return tokens

    def _validate_input(self, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens with shape [B, N, D], received {tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension"
                f"{self.embedding_dim}, "
                f"received {tokens.shape[-1]}"
            )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        number_of_layers: int,
        number_of_heads: int,
        mlp_hidden_dim: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qkv_bias: bool = True,
        layer_norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be greater than zero")

        if number_of_heads <= 0:
            raise ValueError("number_of_heads must be greater than zero")

        if mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be greater than zero")

        if layer_norm_epsilon <= 0.0:
            raise ValueError("layer_norm_epsilon must be greater than zero")

        self.embedding_dim = embedding_dim
        self.number_of_layers = number_of_layers

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embedding_dim=embedding_dim,
                    number_of_heads=number_of_heads,
                    mlp_hidden_dim=mlp_hidden_dim,
                    attention_dropout=attention_dropout,
                    projection_dropout=projection_dropout,
                    mlp_dropout=mlp_dropout,
                    qkv_bias=qkv_bias,
                    layer_norm_epsilon=layer_norm_epsilon,
                )
                for _ in range(number_of_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(
            normalized_shape=embedding_dim,
            eps=layer_norm_epsilon,
        )

    def forward(
        self,
        tokens: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate_input(tokens)

        if not return_attention:
            for block in self.blocks:
                tokens = block(tokens)

            tokens = self.final_norm(tokens)
            return tokens

        attention_maps: list[Tensor] = []

        for block in self.blocks:
            tokens, attention_weights = block(
                tokens,
                return_attention=True,
            )

            attention_maps.append(attention_weights)

        tokens = self.final_norm(tokens)

        stacked_attention_maps = torch.stack(
            attention_maps,
            dim=0,
        )

        return tokens, stacked_attention_maps

    def _validate_input(self, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens with shape [B, N, D], received {tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.embedding_dim}, "
                f"received {tokens.shape[-1]}"
            )
