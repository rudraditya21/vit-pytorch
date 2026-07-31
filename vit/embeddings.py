from __future__ import annotations

import torch
from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embedding_dim: int
    ) -> None:
        super().__init__()

        if image_size <= 0:
            raise ValueError("image_size must be greater than zero")

        if patch_size <= 0:
            raise ValueError("patch_size must be greater than zero")

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by ",
                f"patch_size ({patch_size})",
            )

        if in_channels <= 0:
            raise ValueError("in_channels must be greater than zero")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.grid_size = image_size // patch_size
        self.number_of_patches = self.grid_size * self.grid_size

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, images: Tensor) -> Tensor:
        self._validate_input(images)

        patch_embeddings = self.projection(images)
        patch_embeddings = patch_embeddings.flatten(start_dim=2)
        patch_embeddings = patch_embeddings.transpose(1, 2)

        return patch_embeddings

    def _validate_input(self, images: Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                "Expected images with shape [B, C, H, W], "
                f"received {tuple(images.shape)}"
            )

        _, channels, height, width = images.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {channels}"
            )

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                f"Expected images of size {self.image_size}x{self.image_size}, "
                f"received {height}x{width}"
            )


class ViTEmbeddings(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embedding_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in the range [0, 1), received {dropout}")

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
        )

        self.embedding_dim = embedding_dim
        self.sequence_length = self.patch_embedding.number_of_patches + 1

        self.class_token = nn.Parameter(torch.empty(1, 1, embedding_dim))

        self.position_embeddings = nn.Parameter(
            torch.empty(1, self.sequence_length, embedding_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.class_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, mean=0.0, std=0.02)
        nn.init.trunc_normal_(
            self.patch_embedding.projection.weight, mean=0.0, std=0.02
        )

        if self.patch_embedding.projection.bias is not None:
            nn.init.zeros_(self.patch_embedding.projection.bias)

    def forward(self, images: Tensor) -> Tensor:
        patch_embeddings = self.patch_embedding(images)

        batch_size = patch_embeddings.shape[0]

        class_tokens = self.class_token.expand(batch_size, -1, -1)

        embeddings = torch.cat((class_tokens, patch_embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings
