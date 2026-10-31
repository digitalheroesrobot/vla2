# VLA Course Roadmap

A hands-on path from the foundations of deep learning to a robot that can
follow language instructions. The course builds small educational models
along the way, with each phase introducing ideas used in modern
vision-language-action (VLA) systems.

## Roadmap at a glance

| Phase | Status | Focus | Outcome |
|---|---|---|---|
| 0 | Available | Math and tensor foundations | Understand matrix operations, gradients, broadcasting, and normalization. |
| 1 | Available | Neural networks, vision, and attention | Train an MLP and CNN, implement autograd, and inspect attention. |
| 2 | Available | Language modeling | Build and train a small GPT-style text model. |
| 3 | Available | Vision Transformers | Turn images into patch tokens and train a small ViT. |
| 4 | Available | Vision-language alignment | Train a mini-CLIP and explore shared image-text embeddings. |
| 5 | Available | Vision-language question answering | Fuse visual features and language to answer questions about an image. |

Phases 0–5 are the lessons currently in this repository. Later phases describe
the intended course direction and are not implemented yet.

## What you'll learn

By working through the available lessons, you'll learn how to:

- Use vectors, matrices, and tensors; reason about gradients, broadcasting, and
  normalization.
- Build and train neural networks, implement automatic differentiation, and
  understand how CNNs learn from images.
- Follow how attention relates information across a sequence, then train a
  small GPT-style language model.
- Represent images as patch tokens and build a small Vision Transformer.
- Align image and text representations with contrastive learning, as used in
  CLIP-style models.
- Combine image features and language-question features to answer simple
  questions about generated images.
- Inspect educational-scale implementations and experiment with each part of
  the learning process.

The planned phases will extend these foundations to robot state and control,
learning from demonstrations, action prediction, simulation, and real-robot
deployment. Those topics are roadmap goals, not lessons currently available
in this repository.

## Available lessons

### Phase 0 — Math and tensor foundations

- [Linear algebra and gradients](phase0/1linear_algebra.py)
- [Matrix multiplication benchmarks](phase0/2matmul_benchmark.py)
- [Broadcasting and normalization](phase0/3broadcasting_and_norms.py)

### Phase 1 — Neural networks, vision, and attention

- [MLP from scratch](phase1/1mlp_from_scratch.py)
- [MLP with images](phase1/2mlp_with_photos.py)
- [Autograd from scratch](phase1/3micro_grad.py)
- [CNN](phase1/4cnn.py)
- [Attention from scratch](phase1/5attention_from_scratch.py)

### Phase 2 — Language modeling

- [A tiny GPT](phase2/1gpt_mini.py)

### Phase 3 — Vision Transformers

- [A ViT from scratch](phase3/vit_from_scratch.py)

### Phase 4 — Vision-language alignment

- [Mini-CLIP](phase4/clip_mini.py)

### Phase 5 — Vision-language question answering

- [A mini vision-language QA model](phase5/vlm_mini.py)

## Learning path

The available lessons are standalone scripts, but they build on one another:

1. Learn the tensor operations and gradient math behind neural networks.
2. Train simple models and develop intuition for image features and attention.
3. Build a small language model and a vision transformer.
4. Align image and text representations with contrastive learning.
5. Combine image and question features in a small visual question-answering
   model.
6. Continue toward robot control, demonstrations, action prediction, and
   deployment as the planned phases are added.

These are educational-scale implementations intended to make core ideas
inspectable; they are not production models or a complete VLA system.

## Get started

Use Python with NumPy, PyTorch, Matplotlib, Pillow, and NetworkX:

```bash
python -m pip install numpy torch matplotlib pillow networkx
```

For a specific CUDA version, install the PyTorch build that matches your
system. Run scripts from the repository root, for example:

```bash
python phase0/1linear_algebra.py
python phase1/5attention_from_scratch.py
python phase2/1gpt_mini.py
python phase4/clip_mini.py
python phase5/vlm_mini.py
```

Each lesson runs independently. Most use generated toy data and do not need
downloads; the GPT lesson creates a small text corpus if one is missing.

## Data and generated files

The image-classification and ViT lessons can use images arranged by class:

```text
data/
└── tiny_classify/
    ├── circle/
    ├── square/
    └── triangle/
```

Without a local image dataset, relevant lessons use or generate sample data.
Mini-CLIP trains on generated image-caption pairs, and the Phase 5 QA lesson
generates colored-shape images and questions in memory.

Plots, visualizations, and checkpoints are written to `outputs/`. Some
lessons also generate small datasets under `data/` when they run.
