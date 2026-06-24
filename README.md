# LLM Fundamentals for Engineers

`llm-fundamentals-for-engineers` is an open-source course for Platform Engineers, Cloud Engineers, Kubernetes Engineers, DevOps Engineers, and Software Engineers who want to understand how large language models work from an engineering point of view.

This repository is intentionally written like a technical book, not a slide dump and not a research paper. The goal is to help readers follow the journey of a token through an LLM system, from raw text to generated output.

## Who This Course Is For

- Platform Engineers
- Cloud Engineers
- Kubernetes Engineers
- DevOps Engineers
- Software Engineers

## Who This Course Is Not For

- AI Researchers
- ML Researchers
- Data Scientists looking for proof-heavy treatment

## Course Philosophy

- Teach the journey of a token.
- Reuse one running example across chapters: `The capital of France is`.
- Prefer engineering intuition over formal proofs.
- Explain what the system is doing, why it is doing it, and what engineers should care about.
- Build concepts in sequence so each chapter depends on the previous one.

## What Students Will Learn

- Tokenization
- Embeddings
- Transformer architecture
- Attention
- Hidden states
- Training
- Forward pass
- Backpropagation
- Fine-tuning
- LoRA
- Inference
- Prefill
- Decode
- KV cache

## What This Course Does Not Cover

These topics belong in a later session focused on scaling and production runtime architecture:

- vLLM
- Tensor parallelism
- Pipeline parallelism
- Ray
- Kubernetes deployment patterns
- Multi-GPU
- Multi-node
- NCCL
- Distributed inference

## Repository Layout

```text
README.md

docs/
    00-course-overview.md
    01-introduction.md
    02-tokenization.md
    03-embeddings.md
    04-transformer.md
    05-attention.md
    06-training.md
    07-fine-tuning.md
    08-inference.md
    09-summary.md

assets/
    diagrams/
    figures/
    icons/

slides/

references/
```

## Reading Order

1. [Course Overview](docs/00-course-overview.md)
2. [Introduction](docs/01-introduction.md)
3. [Tokenization](docs/02-tokenization.md)
4. [Embeddings](docs/03-embeddings.md)
5. [Transformer](docs/04-transformer.md)
6. [Attention](docs/05-attention.md)
7. [Training](docs/06-training.md)
8. [Fine-Tuning](docs/07-fine-tuning.md)
9. [Inference](docs/08-inference.md)
10. [Summary](docs/09-summary.md)

## Diagram Standard

All diagrams in `assets/diagrams/` follow the same constraints:

- White background
- Flat, modern visual style
- Large readable fonts
- Minimal clutter
- 16:9 canvas
- PowerPoint-friendly SVG format

## How To Use This Repository

- Read the chapters in order.
- Reuse the SVG diagrams in PowerPoint or internal training material.
- Extend the `slides/` folder if you want to turn the book into a workshop or talk.
- Add more examples only if they reinforce the same token journey rather than fragmenting the story.

## Contributing

Contributions should preserve the current teaching style:

- Write in simple English.
- Explain systems like an engineer teaching another engineer.
- Avoid unnecessary proof-heavy sections.
- Prefer one strong running example over many disconnected examples.
- Keep new diagrams visually consistent with the existing set.
