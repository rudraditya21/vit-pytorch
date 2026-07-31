from __future__ import annotations

import torch

from vit import VisionTransformer


def print_model_summary(model: VisionTransformer) -> None:
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print("=" * 80)
    print("Vision Transformer")
    print("=" * 80)
    print(model)
    print()

    print(f"Total Parameters     : {total_parameters:,}")
    print(f"Trainable Parameters : {trainable_parameters:,}")
    print("=" * 80)


def run_forward_pass(model: VisionTransformer) -> None:
    images = torch.randn(8, 3, 32, 32)

    logits, attention_maps = model(
        images,
        return_attention=True,
    )

    print()
    print("=" * 80)
    print("Forward Pass")
    print("=" * 80)

    print(f"Input Images      : {images.shape}")
    print(f"Output Logits     : {logits.shape}")
    print(f"Attention Maps    : {attention_maps.shape}")

    print()
    print("Per Layer Attention Shape")

    for layer_index, layer_attention in enumerate(attention_maps):
        print(f"Layer {layer_index + 1}: {layer_attention.shape}")


def main() -> None:
    torch.manual_seed(42)

    model = VisionTransformer(
        image_size=32,
        patch_size=4,
        in_channels=3,
        number_of_classes=10,
        embedding_dim=192,
        number_of_layers=6,
        number_of_heads=3,
        mlp_hidden_dim=768,
        embedding_dropout=0.1,
        attention_dropout=0.1,
        projection_dropout=0.1,
        mlp_dropout=0.1,
    )

    print_model_summary(model)

    run_forward_pass(model)


if __name__ == "__main__":
    main()
