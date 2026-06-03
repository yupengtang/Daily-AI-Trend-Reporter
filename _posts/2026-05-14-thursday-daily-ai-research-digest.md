---
layout: post
title: "Daily AI Research Papers - Thursday, May 14, 2026"
date: 2026-05-14
---

Keywords: Low-Rank Adaptation, LoRA, post-training, online serving, base-model deployments, full checkpoint, adapter revisions, distributed training

---

### 1. MinT: Managed Infrastructure for Training and Serving Millions of LLMs

[Read Paper](https://huggingface.co/papers/2605.13779)

MinT is a managed infrastructure system that enables efficient low-rank adaptation training and serving by keeping base models resident and moving lightweight adapter revisions, scaling across multiple dimensions including large model architectures, reduced storage requirements, and distributed policy management.

### 2. MulTaBench: Benchmarking Multimodal Tabular Learning with Text and Image

[Read Paper](https://huggingface.co/papers/2605.10616)

Multimodal tabular learning benchmarks reveal that task-specific embedding tuning improves performance over frozen pretrained embeddings, particularly when modalities provide complementary predictive signals.

### 3. AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation

[Read Paper](https://huggingface.co/papers/2605.13724)

AnyFlow introduces a novel any-step video diffusion distillation framework that improves upon consistency distillation by optimizing full ODE sampling trajectories through flow-map transition learning and backward simulation techniques.

### 4. Training Long-Context Vision-Language Models Effectively with Generalization Beyond 128K Context

[Read Paper](https://huggingface.co/papers/2605.13831)

Long-context continued pre-training enhances vision-language models' ability to handle extended documents while maintaining performance across diverse contexts through strategic data mixture design.

### 5. EVA-Bench: A New End-to-end Framework for Evaluating Voice Agents

[Read Paper](https://huggingface.co/papers/2605.13841)

EVA-Bench presents a comprehensive evaluation framework for voice agents that simulates realistic conversations and measures performance across multiple voice-specific failure modes using novel accuracy and experience metrics.

### 6. Qwen-Image-VAE-2.0 Technical Report

[Read Paper](https://huggingface.co/papers/2605.13565)

Qwen-Image-VAE-2.0 is a high-compression Variational Autoencoder suite that improves reconstruction fidelity and diffusability through enhanced architecture, large-scale training, and semantic alignment strategies.

### 7. Predicting Decisions of AI Agents from Limited Interaction through Text-Tabular Modeling

[Read Paper](https://huggingface.co/papers/2605.12411)

AI agents can predict counterpart decisions in negotiation games by combining tabular features with LLM-based text representations and hidden states from a frozen observer model, outperforming direct prompting methods.

### 8. TrackCraft3R: Repurposing Video Diffusion Transformers for Dense 3D Tracking

[Read Paper](https://huggingface.co/papers/2605.12587)

TrackCraft3R enables efficient dense 3D tracking from monocular video by adapting video diffusion transformers to follow physical points across frames using dual-latent representation and temporal RoPE alignment.

### 9. Edit-Compass & EditReward-Compass: A Unified Benchmark for Image Editing and Reward Modeling

[Read Paper](https://huggingface.co/papers/2605.13062)

Recent image editing models have achieved remarkable progress in instruction following, multimodal understanding, and complex visual editing. However, existing benchmarks often fail to faithfully reflect human judgment, especially for strong frontier models, due to limited task difficulty and coarse-grained evaluation protocols. In parallel, reward models have become increasingly important for RL-based image editing optimization, yet existing reward model benchmarks still rely on unrealistic evaluation settings that deviate from practical RL scenarios. These limitations hinder reliable assessment of both image editing models and reward models. To address these challenges, we introduce Edit-Compass and EditReward-Compass, a unified evaluation suite for image editing and reward modeling. Edit-Compass contains 2,388 carefully annotated instances spanning six progressively challenging task categories, covering capabilities such as world knowledge reasoning, visual reasoning, and multi-image editing. Beyond broad task coverage, Edit-Compass adopts a fine-grained multidimensional evaluation framework based on structured reasoning and carefully designed scoring rubrics. In parallel, EditReward-Compass contains 2,251 preference pairs that simulate realistic reward modeling scenarios during RL optimization.

### 10. Many-Shot CoT-ICL: Making In-Context Learning Truly Learn

[Read Paper](https://huggingface.co/papers/2605.13511)

Many-shot in-context learning for reasoning tasks exhibits different scaling behaviors than non-reasoning tasks, with demonstration ordering and selection significantly impacting performance.
