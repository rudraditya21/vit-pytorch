# Vision Transformer from Scratch

A complete implementation of the **Vision Transformer (ViT)** architecture from **"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"**, implemented entirely from scratch in **PyTorch**.

---

## Features

- Patch Embedding
- Learnable Class Token
- Learnable Positional Embeddings
- Multi-Head Self Attention
- Feed Forward Network (MLP)
- Transformer Encoder Blocks
- Transformer Encoder Stack
- Complete Vision Transformer
- CIFAR-10 Training Pipeline
- AdamW Optimizer
- Linear Warmup + Cosine Learning Rate Scheduler
- Automatic Mixed Precision (CUDA)
- Gradient Clipping
- Model Checkpointing
- Resume Training
- Extensive Unit Tests

---

## Model Architecture

```mermaid
flowchart TD

A[Input Image]
B[Patch Embedding]
C[Flatten Patches]
D[Linear Projection]
E[Add Class Token]
F[Add Positional Embeddings]
G[Transformer Encoder]
H[LayerNorm]
I[CLS Token]
J[Classification Head]
K[Logits]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
G --> H
H --> I
I --> J
J --> K
```

---

## Transformer Encoder

```mermaid
flowchart TD

A[Input Tokens]

B[LayerNorm]
C[Multi-Head Self Attention]
D[Residual Addition]

E[LayerNorm]
F[Feed Forward Network]
G[Residual Addition]

A --> B
B --> C
C --> D
A --> D

D --> E
E --> F
F --> G
D --> G
```

---

## Multi-Head Self Attention

```mermaid
flowchart TD

A[Input]

B[Linear Projection]
C[Q]
D[K]
E[V]

F[Split Heads]

G[Scaled Dot Product]

H[Softmax]

I[Weighted Values]

J[Concatenate Heads]

K[Output Projection]

A --> B

B --> C
B --> D
B --> E

C --> F
D --> F
E --> F

F --> G
G --> H
H --> I
I --> J
J --> K
```

---

## Feed Forward Network

```mermaid
flowchart TD

A[Input]

B[Linear]

C[GELU]

D[Dropout]

E[Linear]

F[Dropout]

G[Output]

A --> B
B --> C
C --> D
D --> E
E --> F
F --> G
```

---

## Training Pipeline

The training pipeline includes

- CIFAR-10 data loading
- Data augmentation
- Data normalization
- AdamW optimizer
- Linear warmup
- Cosine learning-rate decay
- Automatic Mixed Precision (CUDA)
- Gradient clipping
- Checkpoint saving
- Resume training

---

## Running

Install dependencies

```bash
pip install -r requirements.txt
```

Run training

```bash
python main.py
```

---

## Testing

Run the complete test suite

```bash
pytest
```

or

```bash
pytest -v
```

---

## Paper

**An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale**

- **Authors:** Alexey Dosovitskiy et al.
- **Conference:** ICLR 2021

--- 

## References

- Vision Transformer (ViT)
- PyTorch
- CIFAR-10
- AdamW Optimizer
