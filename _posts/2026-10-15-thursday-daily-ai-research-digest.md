---
layout: post
title: "Daily AI Research Papers - Thursday, October 15, 2026"
date: 2026-10-15
---

Keywords: on-policy distillation, token-level supervision, agentic video generation, screenplay, spatial consistency, temporal consistency, scene memory, event memory

---

### 1. Any Time, Any Space: A World-State-Centric Framework for Agentic Video Generation

[Read Paper](https://huggingface.co/papers/2610.20571)

ATAS improves long-form video consistency by compiling screenplays into referenced shot requests, retrieving spatial and temporal evidence from scene and event memories, and conditionally updating memory after generation.

### 2. Beyond Domain-Routed Supervision: Cross-Teacher Guarding for Multi-Teacher On-Policy Distillation

[Read Paper](https://huggingface.co/papers/2610.13521)

Road MOPD reduces gradient conflicts in multi-teacher distillation by using cross-teacher agreement for token-level credit assignment and auxiliary gradients to constrain conflicting updates.

### 3. Player-1: Learning to Play from Synthetic Experience

[Read Paper](https://huggingface.co/papers/2610.20584)

Player-1 improves game control by combining supervised fine-tuning on gameplay trajectories with reinforcement learning across synthetic environments, achieving substantial gains over demonstration-only baselines.

### 4. Doc2Harness: Benchmarking LLMs on Long-Horizon Harness Engineering of Coding Agents

[Read Paper](https://huggingface.co/papers/2610.20344)

Doc2Harness is a deterministic two-stage benchmark that evaluates long-horizon harness engineering by having models build and run harnesses across isolated containers, revealing substantial performance gaps among frontier models.

### 5. More Methods, Better Agents? A Reality Check on Self-Distillation in Agentic Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2610.19789)

Researchers benchmarked ten self-distillation methods against GRPO across multiple agent tasks and model sizes, finding limited consistent gains, hidden training instability, and no reliable cost-to-performance benefit.

### 6. Recovering Reliable Supervision via Critique-Guided On-Policy Distillation

[Read Paper](https://huggingface.co/papers/2610.20572)

CriticOPD improves on-policy distillation by using teacher critiques to restore recoverability, enabling students to revise incorrect reasoning and achieve stronger mathematical reasoning performance.

### 7. Cortex: Consolidating Context into Parametric Memory

[Read Paper](https://huggingface.co/papers/2610.20399)

Cortex consolidates context-induced improvements into mergeable low-rank or structural updates, preserving non-target capabilities across sequential edits and diverse model scales.

### 8. LatentLoop: Scaling Looped MoE Models with Asymmetric Compute Allocation

[Read Paper](https://huggingface.co/papers/2610.19975)

LatentLoop improves looped Mixture-of-Experts by compressing the first loop visit, reallocating saved computation to larger hidden dimensions, and using separate attention parameters, yielding lower validation loss and better benchmark scores.

### 9. ActionOPD: Distilling Decisions, Not Tokens

[Read Paper](https://huggingface.co/papers/2610.19820)

ActionOPD aligns smaller language-model agents with teachers via semantic action-level on-policy distillation, improving out-of-distribution tool-use performance without requiring shared tokenizers or teacher probabilities.

### 10. Seeing, Maintaining, and Learning: The Evolution of Multimodal Agent Memory

[Read Paper](https://huggingface.co/papers/2610.19366)

This work proposes a two-axis framework for multimodal agent memory covering information formation and adaptive management to support lifelong learning.
