---
layout: post
title: "Daily AI Research Papers - Tuesday, February 10, 2026"
date: 2026-02-10
---

Keywords: AI Research Science Benchmark, Anchor Alignment, BigCodeBench, Centroid Alignment, Chain-of-Thought prompting, Fixed-frame Modality Gap Theory, Generator-Critic protocol, HumanEval+

---

### 1. Weak-Driven Learning: How Weak Agents make Strong Agents Stronger

[Read Paper](https://huggingface.co/papers/2602.08222)

WMSS is a post-training paradigm that uses weak model checkpoints to identify and fill learning gaps, enabling continued improvement beyond conventional saturation points in large language models.

### 2. TermiGen: High-Fidelity Environment and Robust Trajectory Synthesis for Terminal Agents

[Read Paper](https://huggingface.co/papers/2602.07274)

TermiGen introduces a pipeline for generating verifiable terminal environments and resilient trajectories to improve open-weight LLMs' ability to execute complex tasks and recover from runtime errors.

### 3. QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining

[Read Paper](https://huggingface.co/papers/2602.07085)

Financial markets are noisy and non-stationary, making alpha mining highly sensitive to noise in backtesting results and sudden market regime shifts. While recent agentic frameworks improve alpha mining automation, they often lack controllable multi-round search and reliable reuse of validated experience. To address these challenges, we propose QuantaAlpha, an evolutionary alpha mining framework that treats each end-to-end mining run as a trajectory and improves factors through trajectory-level mutation and crossover operations. QuantaAlpha localizes suboptimal steps in each trajectory for targeted revision and recombines complementary high-reward segments to reuse effective patterns, enabling structured exploration and refinement across mining iterations. During factor generation, QuantaAlpha enforces semantic consistency across the hypothesis, factor expression, and executable code, while constraining the complexity and redundancy of the generated factor to mitigate crowding. Extensive experiments on the China Securities Index 300 (CSI 300) demonstrate consistent gains over strong baseline models and prior agentic systems. When utilizing GPT-5.2, QuantaAlpha achieves an Information Coefficient (IC) of 0.1501, with an Annualized Rate of Return (ARR) of 27.75% and a Maximum Drawdown (MDD) of 7.98%. Moreover, factors mined on CSI 300 transfer effectively to the China Securities Index 500 (CSI 500) and the Standard & Poor's 500 Index (S&P 500), delivering 160% and 137% cumulative excess return over four years, respectively, which indicates strong robustness of QuantaAlpha under market distribution shifts.

### 4. MOVA: Towards Scalable and Synchronized Video-Audio Generation

[Read Paper](https://huggingface.co/papers/2602.08794)

MOVA is an open-source model that generates synchronized audio-visual content using a Mixture-of-Experts architecture with 32 billion parameters, supporting image-text to video-audio generation tasks.

### 5. Modality Gap-Driven Subspace Alignment Training Paradigm For Multimodal Large Language Models

[Read Paper](https://huggingface.co/papers/2602.07026)

Researchers address the modality gap in multimodal learning by proposing a fixed-frame theory and a training-free alignment method that enables efficient scaling of multimodal models using unpaired data.

### 6. AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents

[Read Paper](https://huggingface.co/papers/2602.06855)

AIRS-Bench presents a comprehensive benchmark suite for evaluating LLM agents across diverse scientific domains, demonstrating current limitations while providing open-source resources for advancement.

### 7. InternAgent-1.5: A Unified Agentic Framework for Long-Horizon Autonomous Scientific Discovery

[Read Paper](https://huggingface.co/papers/2602.08990)

InternAgent-1.5 is a unified system for autonomous scientific discovery that integrates computational modeling and experimental research through coordinated subsystems for generation, verification, and evolution.

### 8. LLaDA2.1: Speeding Up Text Diffusion via Token Editing

[Read Paper](https://huggingface.co/papers/2602.08676)

LLaDA2.1 introduces a novel token-to-token editing approach with speed and quality modes, enhanced through reinforcement learning for improved reasoning and instruction following in large language diffusion models.

### 9. Recurrent-Depth VLA: Implicit Test-Time Compute Scaling of Vision-Language-Action Models via Latent Iterative Reasoning

[Read Paper](https://huggingface.co/papers/2602.07845)

RD-VLA introduces a recurrent architecture for vision-language-action models that adapts computational depth through latent iterative refinement, achieving constant memory usage and improved task success rates.

### 10. RLinf-USER: A Unified and Extensible System for Real-World Online Policy Learning in Embodied AI

[Read Paper](https://huggingface.co/papers/2602.07837)

USER is a unified systems framework that enables scalable, asynchronous online policy learning in physical robots by treating them as first-class hardware resources and supporting diverse learning paradigms including VLA models.
