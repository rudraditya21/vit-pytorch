from __future__ import annotations

from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        number_of_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        if number_of_heads <= 0:
            raise ValueError("number_of_heads must be greater than zero")

        if embedding_dim % number_of_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"number_of_heads ({number_of_heads})"
            )

        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError(
                f"attention_dropout must be in the range [0, 1), received {attention_dropout}"
            )

        if not 0.0 <= projection_dropout < 1.0:
            raise ValueError(
                f"projection_dropout must be in the range [0, 1), received {projection_dropout}"
            )

        self.embedding_dim = embedding_dim
        self.number_of_heads = number_of_heads

        self.head_dim = embedding_dim // number_of_heads
        self.scale = self.head_dim**-0.5

        self.qkv_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim * 3,
            bias=qkv_bias,
        )

        self.attention_dropout = nn.Dropout(attention_dropout)

        self.output_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
        )

        self.projection_dropout = nn.Dropout(projection_dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.qkv_projection.weight)

        if self.qkv_projection.bias is not None:
            nn.init.zeros_(self.qkv_projection.bias)

        nn.init.xavier_uniform_(self.output_projection.weight)

        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        tokens: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate_input(tokens)

        batch_size, sequence_length, _ = tokens.shape

        qkv = self.qkv_projection(tokens)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.number_of_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        attention_scores = queries @ keys.transpose(-2, -1)
        attention_scores = attention_scores * self.scale

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended_values = attention_weights @ values
        attended_values = attended_values.transpose(1, 2)
        attended_values = attended_values.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        output = self.output_projection(attended_values)
        output = self.projection_dropout(output)

        if return_attention:
            return output, attention_weights

        return output

    def _validate_input(self, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens with shape [B, N, D], received {tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected token embedding dimension "
                f"{self.embedding_dim}, "
                f"received {tokens.shape[-1]}"
            )
