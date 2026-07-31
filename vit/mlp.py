from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero")

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in the range [0, 1), received {dropout}")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(
            in_features=embedding_dim,
            out_features=hidden_dim,
        )

        self.activation = nn.GELU()

        self.hidden_dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(
            in_features=hidden_dim,
            out_features=embedding_dim,
        )

        self.output_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens: Tensor) -> Tensor:
        self._validate_input(tokens)

        tokens = self.input_projection(tokens)
        tokens = self.activation(tokens)
        tokens = self.hidden_dropout(tokens)

        tokens = self.output_projection(tokens)
        tokens = self.output_dropout(tokens)

        return tokens

    def _validate_input(self, tokens: Tensor) -> None:
        if tokens.ndim != 3:
            raise ValueError(
                f"Expected tokens with shape [B, N, D], received {tuple(tokens.shape)}"
            )

        if tokens.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {tokens.shape[-1]}"
            )
