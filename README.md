# LLM Fundamentals for Engineers

`llm-fundamentals-for-engineers` is a practical learning repository for software engineers, platform engineers, DevOps engineers, SREs, cloud engineers, Kubernetes engineers, and infrastructure engineers who want to understand how modern LLM systems work. It explains the path from core model concepts to training, inference, serving, and production operations without assuming an ML background.

---

## Why this repository?

LLMs are no longer isolated research artifacts. They are becoming part of production systems that engineers must deploy, scale, secure, observe, and troubleshoot.

If you already understand containers, Kubernetes, APIs, distributed systems, and cloud platforms, you already have much of the systems intuition needed to reason about LLM platforms. What is usually missing is the model-side context: what a token is, why GPUs matter, what happens during inference, why KV cache exists, what fine-tuning changes, and how serving stacks behave under load.

This repository focuses on those gaps.

The goal is not to turn platform engineers into ML researchers. The goal is to help engineers understand:

- what problem each LLM component solves
- where it fits in the lifecycle from training to serving
- what tradeoffs show up in production
- how model behavior connects to infrastructure decisions

---

## What you'll learn

### Part 0 — Introduction

- What is an LLM
- AI ecosystem
- Why LLMs matter

### Part 1 — LLM Fundamentals

- Tokens
- Embeddings
- Transformer
- Attention
- Feed Forward Network
- Positional Encoding
- Training
- Inference

### Part 2 — Production AI Systems

- GPUs
- Model Storage
- Serving
- KV Cache
- LoRA
- Quantization
- Distributed Inference
- Fine Tuning
- RAG
- AI Gateways
- Guardrails

### Part 3 — LLM on Kubernetes

- Model deployment
- vLLM
- KServe
- Triton
- Ray
- Multi-GPU
- Multi-node
- Scheduling
- Autoscaling
- Monitoring

---

## Repository Structure

```text
/
README.md
docs/
  00-introduction/
    README.md
  01-text-representation/
    README.md
  02-positional-encoding/
    README.md
  03-transformer-architecture/
    README.md
  04-self-attention/
    README.md
  05-feed-forward-and-residual/
    README.md
  06-multi-head-attention/
    README.md
  07-one-complete-transformer-forward-pass/
    README.md
  08-training-foundation-models/
    README.md
  09-from-foundation-models-to-chat-models/
    README.md
  10-parameter-efficient-fine-tuning/
    README.md
  11-inference/
    README.md
  12-kv-cache/
    README.md
  13-quantization/
    README.md
  14-model-serving/
    README.md
  15-model-storage/
    README.md
  16-distributed-inference/
    README.md
  17-llms-on-kubernetes/
    README.md
  18-building-ai-applications/
    README.md
  19-production-ai-platform-architecture/
    README.md
  00-course-overview.md          # legacy redirect
  01-introduction.md            # legacy redirect
  02-tokenization.md            # legacy redirect
  03-embeddings.md              # legacy redirect
  04-transformer.md             # legacy redirect
  05-attention.md               # legacy redirect
  06-training.md                # legacy redirect
  07-fine-tuning.md             # legacy redirect
  08-inference.md               # legacy redirect
  09-summary.md                 # legacy redirect
slides/
  README.md
references/
  README.md
assets/
  diagrams/
  figures/
  icons/
```

---

## Learning Path

| Chapter | Topic | Estimated Time |
| --- | --- | --- |
| 00 | [Introduction to Large Language Models](docs/00-introduction/README.md) | 10 min |
| 01 | [From Text to Numbers: Tokens and Embeddings](docs/01-text-representation/README.md) | 15 min |
| 02 | [Understanding Context with Positional Encoding](docs/02-positional-encoding/README.md) | 15 min |
| 03 | [Inside a Transformer](docs/03-transformer-architecture/README.md) | 12 min |
| 04 | [Self-Attention: The Core of the Transformer](docs/04-self-attention/README.md) | 20 min |
| 05 | [Feed Forward Networks, Residual Connections, and Layer Normalization](docs/05-feed-forward-and-residual/README.md) | 18 min |
| 06 | [Multi-Head Attention](docs/06-multi-head-attention/README.md) | 18 min |
| 07 | [One Complete Transformer Forward Pass](docs/07-one-complete-transformer-forward-pass/README.md) | 18 min |
| 08 | [Training Foundation Models](docs/08-training-foundation-models/README.md) | 20 min |
| 09 | [From Foundation Models to Chat Models](docs/09-from-foundation-models-to-chat-models/README.md) | 18 min |
| 10 | [Parameter-Efficient Fine-Tuning](docs/10-parameter-efficient-fine-tuning/README.md) | 18 min |
| 11 | [Inference](docs/11-inference/README.md) | 18 min |
| 12 | [KV Cache](docs/12-kv-cache/README.md) | 16 min |
| 13 | [Quantization](docs/13-quantization/README.md) | 16 min |
| 14 | [Model Serving](docs/14-model-serving/README.md) | 18 min |
| 15 | [Model Storage](docs/15-model-storage/README.md) | 15 min |
| 16 | [Distributed Inference](docs/16-distributed-inference/README.md) | 18 min |
| 17 | [LLMs on Kubernetes](docs/17-llms-on-kubernetes/README.md) | 18 min |
| 18 | [Building AI Applications](docs/18-building-ai-applications/README.md) | 20 min |
| 19 | [Production AI Platform Architecture](docs/19-production-ai-platform-architecture/README.md) | 20 min |

Note: the older flat files under `docs/*.md` are kept only as compatibility redirects. The canonical course now lives in the nested chapter directories under `docs/*/README.md`.

---

## Guiding Principles

- Engineering first
- Visual explanations
- Minimal math
- Production focused
- Kubernetes oriented
- Cloud native
- Vendor neutral

---

## Who should read this?

✅ Kubernetes engineers

✅ Platform engineers

✅ DevOps engineers

✅ Software engineers

✅ Cloud architects

✅ Infrastructure engineers

---

## Prerequisites

Readers should already know:

- Linux
- Docker
- Kubernetes basics
- REST APIs
- Basic networking

No ML experience required.

---

## Contributing

When contributing new material:

- Keep diagrams in Mermaid.
- Prefer concise explanations.
- One concept per chapter.
- Keep production focus.
- Avoid unnecessary mathematics.

---

## References

References will be added as chapters are completed.
