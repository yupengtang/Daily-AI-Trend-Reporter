---
layout: post
title: "Daily AI Research Papers - Monday, November 09, 2026"
date: 2026-11-09
---

Keywords: feed-forward 3DGS, 3D Gaussian Splatting, Gaussian attributes, radial-distance-normalized space, Gaussian centers, Gaussian-scale targets, Gaussian Spacing Normalization, GSNorm

---

### 1. SplatAnything: Gaussian Spacing Normalization for Feed-forward 3DGS across Arbitrary Scenes

[Read Paper](https://huggingface.co/papers/2611.08264)

SplatAnything applies Gaussian Spacing Normalization to canonicalize scene-level Gaussian spacing, yielding more consistent supervision and improved feed-forward 3D Gaussian Splatting under varying scene scales and viewpoints.

### 2. P^2-MOPD: Accelerating Multi-Teacher On-Policy Distillation via Parameter Priors

[Read Paper](https://huggingface.co/papers/2611.03953)

P^2-MOPD jointly optimizes mixing coefficients and student weights under an on-policy distillation objective, enabling faster and more balanced recovery than MOPD by treating experts as parameter priors while keeping the student trainable.

### 3. LongBanana: An Expert-Verified Benchmark for Long-Context Multi-Reference Image Synthesis

[Read Paper](https://huggingface.co/papers/2611.06856)

LongBanana is an expert-verified benchmark and agent-harness framework for evaluating and improving long-context multi-reference image synthesis through graph-grounded checklists and an RPR workflow.

### 4. PCBRouteAgent: How Close Are LLM Agents to Becoming PCB Routing Engineers?

[Read Paper](https://huggingface.co/papers/2611.08175)

Automating printed circuit board (PCB) routing requires coordinating board-level connectivity decisions with geometric execution under strict design rules. Although recent agents can interact with native routing tools, reliable routing on complex boards remains challenging. We present PCBRouteAgent, a multimodal PCB routing agent inspired by how human engineers coordinate global strategy with local execution. A Global Routing Planner combines board images and structured electrical and geometric information to specify connection topology, routing order, corridors, and layer transitions. A Detailed Routing Executor realizes these guides through KiCad's native routing engine and performs bounded local recovery. Unresolved failures are returned as structured feedback, enabling targeted guide revisions while preserving accepted routes. We evaluate PCBRouteAgent on 247 synthetic and real-world boards from four PCBWorld-Bench test splits. It achieves clean-pass rates of 100%, 94.9%, 100%, and 40% on D2, D3-A, D3-B, and D3-C, respectively. On the three real-world subsets, PCBRouteAgent outperforms the best evaluated baseline in clean-pass rate by 13.9, 36, and 10 percentage points, respectively.

### 5. Confucius4-R2T2: A Stable-Prefix Streaming ASR Model for Real Real-Time Transcription

[Read Paper](https://huggingface.co/papers/2611.06043)

A full-context speech recognizer is adapted to emit immutable streaming transcripts with near full-context accuracy and low latency by learning longest stable prefix labels.

### 6. Beneath the Surface: What We Overlook When We Talk About Agent Evolution

[Read Paper](https://huggingface.co/papers/2611.07624)

Agent harness evolution can improve benchmark performance without weight updates, but gains are unreliable, costly, and transfer inconsistently, motivating preservation-aware evaluation and EvoAudit revisions.

### 7. LLMs Don't Know What They Know: Benchmarking Temporal Self-Knowledge for Financial Backtesting

[Read Paper](https://huggingface.co/papers/2611.06808)

A benchmark of dated U.S. financial facts shows that large language models often understate their knowledge and that common safeguards cannot reliably certify that historical backtests are free of information leakage.

### 8. Fast and Exact Relative-Position Attention via Structure-Aware Kernels

[Read Paper](https://huggingface.co/papers/2611.01182)

An exact tiled attention algorithm integrates additive relative-position terms into FlashAttention-style computation, reducing memory and compute overhead and accelerating long-context relative-position transformers.

### 9. EvoWorld: Towards Out-of-Sight World Evolution in Generative Video Models

[Read Paper](https://huggingface.co/papers/2611.06748)

EvoWorld combines a vision-language model that maintains and reasons over world state with a segment-autoregressive video renderer to improve event-conditioned causal evolution and revisit consistency in generative video world simulators.

### 10. Video-Coder: Reinforcing Agentic Coding for Video Generation

[Read Paper](https://huggingface.co/papers/2611.09894)

Video-Coder is a code-driven video generation agent that writes and executes code to build structured 3D draft videos, then uses supervised fine-tuning and reinforcement learning with execution and video rewards to improve controllable video generation.
