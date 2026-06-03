---
layout: post
title: "Daily AI Research Papers - Wednesday, December 24, 2025"
date: 2025-12-24
---

Keywords: ADR-Bench, Atomic Capabilities, Bottom-up Policy Optimization, BrowseComp, Checklist-style Judger, Data Synthesis Strategy, Deep Research, EvolveLab

---

### 1. SemanticGen: Video Generation in Semantic Space

[Read Paper](https://huggingface.co/papers/2512.20619)

SemanticGen addresses slow convergence and computational costs in video generation by using a two-stage diffusion model approach that first generates semantic features and then VAE latents, leading to faster convergence and high-quality results.

### 2. Step-DeepResearch Technical Report

[Read Paper](https://huggingface.co/papers/2512.20491)

Step-DeepResearch, an end-to-end agent enhanced with a data synthesis strategy and progressive training, achieves expert-level capabilities in deep research scenarios, outperforming established models.

### 3. Bottom-up Policy Optimization: Your Language Model Policy Secretly Contains Internal Policies

[Read Paper](https://huggingface.co/papers/2512.19673)

The paper decomposes the policy of large language models into internal layer and modular policies, revealing distinct reasoning patterns across layers and proposing Bottom-up Policy Optimization to enhance performance on complex reasoning tasks.

### 4. LongVideoAgent: Multi-Agent Reasoning with Long Videos

[Read Paper](https://huggingface.co/papers/2512.20618)

A multi-agent framework, involving a master LLM, grounding agent, and vision agent, enhances long-video QA by improving temporal grounding and leveraging visual and textual data.

### 5. SpatialTree: How Spatial Abilities Branch Out in MLLMs

[Read Paper](https://huggingface.co/papers/2512.20617)

SpatialTree, a cognitive-science-inspired hierarchy, evaluates and improves spatial abilities in MLLMs across multiple levels, revealing transfer dynamics and proposing an auto-think strategy for consistent performance enhancement.

### 6. Reinforcement Learning for Self-Improving Agent with Skill Library

[Read Paper](https://huggingface.co/papers/2512.17102)

A novel RL framework, SAGE, enhances LLM-based agents' self-improvement capabilities by systematically incorporating skills from a skill library, leading to better performance and efficiency in new environments.

### 7. MemEvolve: Meta-Evolution of Agent Memory Systems

[Read Paper](https://huggingface.co/papers/2512.18746)

MemEvolve, a meta-evolutionary framework, enhances self-evolving memory systems by jointly evolving agents' experiential knowledge and memory architecture, leading to improved performance and generalization.

### 8. SAM Audio: Segment Anything in Audio

[Read Paper](https://huggingface.co/papers/2512.18099)

SAM Audio, a diffusion transformer-based foundation model, achieves superior performance in general audio separation using unified text, visual, and temporal span prompts across various audio types.

### 9. INTELLECT-3: Technical Report

[Read Paper](https://huggingface.co/papers/2512.16144)

INTELLECT-3, a large Mixture-of-Experts model trained with reinforcement learning, achieves top performance across various benchmarks and is supported by an open-source RL infrastructure framework.

### 10. C2LLM Technical Report: A New Frontier in Code Retrieval via Adaptive Cross-Attention Pooling

[Read Paper](https://huggingface.co/papers/2512.21332)

We present C2LLM - Contrastive Code Large Language Models, a family of code embedding models in both 0.5B and 7B sizes. Building upon Qwen-2.5-Coder backbones, C2LLM adopts a Pooling by Multihead Attention (PMA) module for generating sequence embedding from token embeddings, effectively 1) utilizing the LLM's causal representations acquired during pretraining, while also 2) being able to aggregate information from all tokens in the sequence, breaking the information bottleneck in EOS-based sequence embeddings, and 3) supporting flexible adaptation of embedding dimension, serving as an alternative to MRL. Trained on three million publicly available data, C2LLM models set new records on MTEB-Code among models of similar sizes, with C2LLM-7B ranking 1st on the overall leaderboard.
