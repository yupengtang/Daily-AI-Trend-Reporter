---
layout: post
title: "Daily AI Research Papers - Thursday, June 05, 2025"
date: 2025-06-05
---

Keywords: 3D point-cloud sequences, AIME, AMD RDNA3, AmbiK, CASS, CASS-Bench, CUDA, Chain-of-Thought

---

### 1. MiMo-VL Technical Report

[Read Paper](https://huggingface.co/papers/2506.03569)

MiMo-VL-7B-SFT and MiMo-VL-7B-RL provide state-of-the-art general visual understanding and multimodal reasoning through four-stage pre-training and Mixed On-policy Reinforcement Learning, outperforming models with up to 78B parameters.

### 2. OpenThoughts: Data Recipes for Reasoning Models

[Read Paper](https://huggingface.co/papers/2506.04178)

The OpenThoughts project created open-source datasets leading to reasoning models that match or exceed state-of-the-art benchmarks in math, code, and science.

### 3. Advancing Multimodal Reasoning: From Optimized Cold Start to Staged
  Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2506.04207)

Inspired by the remarkable reasoning capabilities of Deepseek-R1 in complex
textual tasks, many works attempt to incentivize similar capabilities in
Multimodal Large Language Models (MLLMs) by directly applying reinforcement
learning (RL). However, they still struggle to activate complex reasoning. In
this paper, rather than examining multimodal RL in isolation, we delve into
current training pipelines and identify three crucial phenomena: 1) Effective
cold start initialization is critical for enhancing MLLM reasoning.
Intriguingly, we find that initializing with carefully selected text data alone
can lead to performance surpassing many recent multimodal reasoning models,
even before multimodal RL. 2) Standard GRPO applied to multimodal RL suffers
from gradient stagnation, which degrades training stability and performance. 3)
Subsequent text-only RL training, following the multimodal RL phase, further
enhances multimodal reasoning. This staged training approach effectively
balances perceptual grounding and cognitive reasoning development. By
incorporating the above insights and addressing multimodal RL issues, we
introduce ReVisual-R1, achieving a new state-of-the-art among open-source 7B
MLLMs on challenging benchmarks including MathVerse, MathVision, WeMath,
LogicVista, DynaMath, and challenging AIME2024 and AIME2025.

### 4. AmbiK: Dataset of Ambiguous Tasks in Kitchen Environment

[Read Paper](https://huggingface.co/papers/2506.04089)

AmbiK, a textual dataset of ambiguous instructions for kitchen robots, enables unified comparison of ambiguity detection methods.

### 5. CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark

[Read Paper](https://huggingface.co/papers/2505.16968)

CASS is a dataset and model suite for GPU code transpilation at both source and assembly levels, achieving high accuracy and performance matching with native code.

### 6. SuperWriter: Reflection-Driven Long-Form Generation with Large Language
  Models

[Read Paper](https://huggingface.co/papers/2506.04180)

SuperWriter-Agent enhances long-form text generation by integrating structured planning and refinement, achieving top performance with a 7B model and hierarchical Direct Preference Optimization.

### 7. A Controllable Examination for Long-Context Language Models

[Read Paper](https://huggingface.co/papers/2506.02921)

LongBioBench is a new benchmark using artificially generated biographies to evaluate long-context language models across understanding, reasoning, and trustworthiness dimensions, addressing limitations in existing frameworks.

### 8. MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in
  Videos

[Read Paper](https://huggingface.co/papers/2506.04141)

A new benchmark, MMR-V, is proposed to challenge multimodal large language models with long-range, multi-frame reasoning and hidden information processing in videos, revealing their limitations and inspiring further research.

### 9. Voyager: Long-Range and World-Consistent Video Diffusion for Explorable
  3D Scene Generation

[Read Paper](https://huggingface.co/papers/2506.04225)

Voyager is a video diffusion framework that generates world-consistent 3D point-cloud sequences from a single image, enabling long-range, consistent 3D scene exploration with user-defined camera paths.

### 10. Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis

[Read Paper](https://huggingface.co/papers/2506.04142)

A method called shortcut neuron patching identifies and suppresses shortcut neurons in language models to mitigate data contamination issues in trustworthy evaluations.
