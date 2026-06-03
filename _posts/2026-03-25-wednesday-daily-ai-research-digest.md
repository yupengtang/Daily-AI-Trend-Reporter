---
layout: post
title: "Daily AI Research Papers - Wednesday, March 25, 2026"
date: 2026-03-25
---

Keywords: reinforcement learning, GRPO, diffusion-based framework, autoregressive decoding, parallel diffusion denoising, block-wise diffusion decoder, uncertainty-driven curriculum learning, inverse rendering perspective

---

### 1. MinerU-Diffusion: Rethinking Document OCR as Inverse Rendering via Diffusion Decoding

[Read Paper](https://huggingface.co/papers/2603.22458)

MinerU-Diffusion is a diffusion-based framework that replaces autoregressive decoding with parallel diffusion denoising for document OCR, improving robustness and decoding speed.

### 2. WildWorld: A Large-Scale Dataset for Dynamic World Modeling with Actions and Explicit State toward Generative ARPG

[Read Paper](https://huggingface.co/papers/2603.23497)

WildWorld is a large-scale dataset for action-conditioned world modeling that provides explicit state annotations from a photorealistic game, enabling better understanding of latent-state dynamics and long-horizon consistency.

### 3. SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning

[Read Paper](https://huggingface.co/papers/2603.23483)

SpecEyes accelerates agentic multimodal large language models by using a lightweight speculative planner with cognitive gating and heterogeneous parallel processing to reduce latency and improve throughput.

### 4. From Static Templates to Dynamic Runtime Graphs: A Survey of Workflow Optimization for LLM Agents

[Read Paper](https://huggingface.co/papers/2603.22386)

LLM-based systems use executable workflows that interleave various computational components, with recent approaches organized by workflow structure determination timing and optimization dimensions.

### 5. DA-Flow: Degradation-Aware Optical Flow Estimation with Diffusion Models

[Read Paper](https://huggingface.co/papers/2603.23499)

Optical flow models trained on high-quality data often degrade severely when confronted with real-world corruptions such as blur, noise, and compression artifacts. To overcome this limitation, we formulate Degradation-Aware Optical Flow, a new task targeting accurate dense correspondence estimation from real-world corrupted videos. Our key insight is that the intermediate representations of image restoration diffusion models are inherently corruption-aware but lack temporal awareness. To address this limitation, we lift the model to attend across adjacent frames via full spatio-temporal attention, and empirically demonstrate that the resulting features exhibit zero-shot correspondence capabilities. Based on this finding, we present DA-Flow, a hybrid architecture that fuses these diffusion features with convolutional features within an iterative refinement framework. DA-Flow substantially outperforms existing optical flow methods under severe degradation across multiple benchmarks.

### 6. SIMART: Decomposing Monolithic Meshes into Sim-ready Articulated Assets via MLLM

[Read Paper](https://huggingface.co/papers/2603.23386)

A unified multimodal large language model framework called SIMART is proposed for generating articulated 3D assets with reduced tokenization overhead and improved simulation readiness.

### 7. PEARL: Personalized Streaming Video Understanding Model

[Read Paper](https://huggingface.co/papers/2603.20422)

Personalized streaming video understanding addresses real-time visual input processing with precise temporal annotations, enabling interactive AI assistants through a new benchmark and plug-and-play strategy.

### 8. UniGRPO: Unified Policy Optimization for Reasoning-Driven Visual Generation

[Read Paper](https://huggingface.co/papers/2603.23500)

A unified reinforcement learning framework is proposed for interleaved text and image generation, using GRPO and FlowGRPO with modifications to enable scalable multi-round generation.

### 9. RealMaster: Lifting Rendered Scenes into Photorealistic Video

[Read Paper](https://huggingface.co/papers/2603.23462)

RealMaster combines video diffusion models with 3D engine outputs to generate photorealistic videos that maintain geometric accuracy and scene consistency through paired training and IC-LoRA distillation.

### 10. Rethinking Token-Level Policy Optimization for Multimodal Chain-of-Thought

[Read Paper](https://huggingface.co/papers/2603.22847)

Researchers developed a token-level reinforcement learning method called PEPO that improves multimodal chain-of-thought reasoning by distinguishing visual grounding from inference through perception-exploration policy optimization.
