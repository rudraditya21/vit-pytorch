from __future__ import annotations

from torch import Tensor, nn

from vit.embeddings import ViTEmbeddings
from vit.encoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        number_of_classes: int,
        embedding_dim: int,
        number_of_layers: int,
        number_of_heads: int,
        mlp_hidden_dim: int,
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        qkv_bias: bool = True,
        layer_norm_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if number_of_classes <= 0:
            raise ValueError("number_of_classes must be greater than zero")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.number_of_classes = number_of_classes
        self.embedding_dim = embedding_dim

        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim,
            dropout=embedding_dropout,
        )

        self.encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
            mlp_dropout=mlp_dropout,
            qkv_bias=qkv_bias,
            layer_norm_epsilon=layer_norm_epsilon,
        )

        self.classification_head = nn.Linear(
            in_features=embedding_dim,
            out_features=number_of_classes,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.classification_head.weight)

        if self.classification_head.bias is not None:
            nn.init.zeros_(self.classification_head.bias)

    def forward(
        self,
        images: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        tokens = self.embeddings(images)

        if return_attention:
            encoded_tokens, attention_maps = self.encoder(
                tokens,
                return_attention=True,
            )
        else:
            encoded_tokens = self.encoder(tokens)

        class_token = encoded_tokens[:, 0]

        logits = self.classification_head(class_token)

        if return_attention:
            return logits, attention_maps

        return logits
