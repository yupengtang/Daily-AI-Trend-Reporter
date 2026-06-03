---
layout: post
title: "Daily AI Research Papers - Friday, March 27, 2026"
date: 2026-03-27
---

Keywords: DiT blocks, Diffusion Transformers, FFE dataset, FFE-Bench, KV cache compression, LTX-2, LoRA, Memory Interleaving

---

### 1. Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale

[Read Paper](https://huggingface.co/papers/2603.25040)

Intern-S1-Pro is a one-trillion-parameter scientific multimodal foundation model that enhances general and scientific capabilities through advanced agent functionalities and specialized task mastery across multiple scientific disciplines.

### 2. PixelSmile: Toward Fine-Grained Facial Expression Editing

[Read Paper](https://huggingface.co/papers/2603.25728)

A diffusion framework called PixelSmile is introduced that disentangles facial expression semantics through symmetric joint training and contrastive learning to enable precise, controllable, and fine-grained expression editing with robust identity preservation.

### 3. Calibri: Enhancing Diffusion Transformers via Parameter-Efficient Calibration

[Read Paper](https://huggingface.co/papers/2603.24800)

Diffusion Transformers can be enhanced through a parameter-efficient calibration approach that improves generative quality while reducing inference steps.

### 4. Voxtral TTS

[Read Paper](https://huggingface.co/papers/2603.25551)

Voxtral TTS is a multilingual text-to-speech model that generates natural speech from short reference audio using a hybrid architecture combining semantic token generation and flow-matching for acoustic tokens.

### 5. RealRestorer: Towards Generalizable Real-World Image Restoration with Large-Scale Image Editing Models

[Read Paper](https://huggingface.co/papers/2603.25502)

A large-scale dataset and open-source model are developed to improve image restoration performance and close the gap with closed-source alternatives, with a dedicated benchmark for real-world degradation evaluation.

### 6. MSA: Memory Sparse Attention for Efficient End-to-End Memory Model Scaling to 100M Tokens

[Read Paper](https://huggingface.co/papers/2603.23516)

Memory Sparse Attention (MSA) enables large language models to process extremely long contexts with linear complexity and high efficiency through innovations like sparse attention and document-wise RoPE.

### 7. MACRO: Advancing Multi-Reference Image Generation with Structured Long-Context Data

[Read Paper](https://huggingface.co/papers/2603.25319)

A large-scale dataset and benchmark are introduced to address limitations in multi-reference image generation by providing structured long-context supervision and standardized evaluation protocols.

### 8. SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks

[Read Paper](https://huggingface.co/papers/2603.24755)

Software development is iterative, yet agentic coding benchmarks overwhelmingly evaluate single-shot solutions against complete specifications. Code can pass the test suite but become progressively harder to extend. Recent iterative benchmarks attempt to close this gap, but constrain the agent's design decisions too tightly to faithfully measure how code quality shapes future extensions. We introduce SlopCodeBench, a language-agnostic benchmark comprising 20 problems and 93 checkpoints, in which agents repeatedly extend their own prior solutions under evolving specifications that force architectural decisions without prescribing internal structure. We track two trajectory-level quality signals: verbosity, the fraction of redundant or duplicated code, and structural erosion, the share of complexity mass concentrated in high-complexity functions. No agent solves any problem end-to-end across 11 models; the highest checkpoint solve rate is 17.2%. Quality degrades steadily: erosion rises in 80% of trajectories and verbosity in 89.8%. Against 48 open-source Python repositories, agent code is 2.2x more verbose and markedly more eroded. Tracking 20 of those repositories over time shows that human code stays flat, while agent code deteriorates with each iteration. A prompt-intervention study shows that initial quality can be improved, but it does not halt degradation. These results demonstrate that pass-rate benchmarks systematically undermeasure extension robustness, and that current agents lack the design discipline iterative software development demands.

### 9. AVControl: Efficient Framework for Training Audio-Visual Controls

[Read Paper](https://huggingface.co/papers/2603.24793)

AVControl enables efficient, modular audio-visual generation by training control modalities as separate LoRA adapters on a parallel canvas within LTX-2, achieving superior performance on diverse control tasks while requiring minimal computational resources.

### 10. Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

[Read Paper](https://huggingface.co/papers/2603.25562)

On-policy distillation for large language models faces challenges in long-horizon settings due to token-level signal fragility, which is addressed through improved estimation methods and implementation techniques.
