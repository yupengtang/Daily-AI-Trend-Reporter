---
layout: post
title: "Daily AI Research Papers - Tuesday, January 06, 2026"
date: 2026-01-06
---

Keywords: 256K-token context window, AutoRegressive, FlashAttention, Group Relative Policy Optimization, IDBench-V, Identity-Anchored Video Synthesizer, Identity-Coherence Reinforcement Learning, Image Face Swapping

---

### 1. Recursive Language Models

[Read Paper](https://huggingface.co/papers/2512.24601)

We study allowing large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. We propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment and allows the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. We find that RLMs successfully handle inputs up to two orders of magnitude beyond model context windows and, even for shorter prompts, dramatically outperform the quality of base LLMs and common long-context scaffolds across four diverse long-context tasks, while having comparable (or cheaper) cost per query.

### 2. K-EXAONE Technical Report

[Read Paper](https://huggingface.co/papers/2601.01739)

K-EXAONE is a multilingual language model with a Mixture-of-Experts architecture that achieves competitive performance on various benchmarks while supporting multiple languages and long-context windows.

### 3. Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits

[Read Paper](https://huggingface.co/papers/2512.20578)

Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision.

### 4. NextFlow: Unified Sequential Modeling Activates Multimodal Understanding and Generation

[Read Paper](https://huggingface.co/papers/2601.02204)

NextFlow is a unified decoder-only autoregressive transformer that processes interleaved text-image tokens, enabling fast multimodal generation through novel next-token and next-scale prediction strategies.

### 5. DreamID-V:Bridging the Image-to-Video Gap for High-Fidelity Face Swapping via Diffusion Transformer

[Read Paper](https://huggingface.co/papers/2601.01425)

A novel video face swapping framework combines image face swapping techniques with diffusion transformers and curriculum learning to achieve superior identity preservation and visual realism.

### 6. SimpleMem: Efficient Lifelong Memory for LLM Agents

[Read Paper](https://huggingface.co/papers/2601.02553)

To support reliable long-term interaction in complex environments, LLM agents require memory systems that efficiently manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which applies entropy-aware filtering to distill unstructured interactions into compact, multi-view indexed memory units; (2) Recursive Memory Consolidation, an asynchronous process that integrates related units into higher-level abstract representations to reduce redundancy; and (3) Adaptive Query-Aware Retrieval, which dynamically adjusts retrieval scope based on query complexity to construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.

### 7. InfiniteVGGT: Visual Geometry Grounded Transformer for Endless Streams

[Read Paper](https://huggingface.co/papers/2601.02281)

InfiniteVGGT enables continuous 3D visual geometry understanding through a causal transformer with adaptive memory management, outperforming existing streaming methods in long-term stability while introducing a new benchmark for extended evaluation.

### 8. VAR RL Done Right: Tackling Asynchronous Policy Conflicts in Visual Autoregressive Generation

[Read Paper](https://huggingface.co/papers/2601.02256)

Visual autoregressive models face training instability due to asynchronous policy conflicts, which are addressed through a novel framework enhancing group relative policy optimization with intermediate rewards, dynamic time-step reweighting, and mask propagation algorithms.

### 9. VINO: A Unified Visual Generator with Interleaved OmniModal Context

[Read Paper](https://huggingface.co/papers/2601.02358)

VINO is a unified visual generator that uses a shared diffusion backbone with multimodal inputs to perform image and video generation and editing tasks.

### 10. GARDO: Reinforcing Diffusion Models without Reward Hacking

[Read Paper](https://huggingface.co/papers/2512.24138)

Online reinforcement learning for diffusion model fine-tuning suffers from reward hacking due to proxy reward mismatches, which GARDO addresses through selective regularization, adaptive reference updates, and diversity-aware reward amplification.
