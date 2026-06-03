---
layout: post
title: "Daily AI Research Papers - Wednesday, January 14, 2026"
date: 2026-01-14
---

Keywords: 3D-aware features, Feature Merger, GitHub, IoU, MUSt3R, Mixture-of-Experts, Positive IoU, RL training

---

### 1. MemGovern: Enhancing Code Agents through Learning from Governed Human Experiences

[Read Paper](https://huggingface.co/papers/2601.06789)

MemGovern framework transforms unstructured GitHub data into structured experiential memory for autonomous software engineering agents, improving bug resolution rates through enhanced experience retrieval.

### 2. Motion Attribution for Video Generation

[Read Paper](https://huggingface.co/papers/2601.08828)

Motive is a gradient-based data attribution framework that identifies influential video clips for motion improvement in text-to-video models through motion-weighted loss masking.

### 3. Solar Open Technical Report

[Read Paper](https://huggingface.co/papers/2601.07022)

Solar Open presents a 102B-parameter bilingual Mixture-of-Experts language model that addresses data scarcity in underserved languages through synthetic data generation, progressive curriculum coordination, and scalable reinforcement learning optimization.

### 4. Ministral 3

[Read Paper](https://huggingface.co/papers/2601.08584)

The Ministral 3 series consists of parameter-efficient dense language models with three sizes (3B, 8B, 14B) and three variants per size, trained using cascade distillation for compute-constrained applications.

### 5. KnowMe-Bench: Benchmarking Person Understanding for Lifelong Digital Companions

[Read Paper](https://huggingface.co/papers/2601.04745)

Long-horizon memory benchmarks based on autobiographical narratives evaluate models' ability to infer stable motivations and decision principles through evidence-linked questions spanning factual recall, subjective state attribution, and principle-level reasoning.

### 6. ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking

[Read Paper](https://huggingface.co/papers/2601.06487)

Reinforcement learning for large language model agents suffers from discrimination collapse in open-ended tasks due to pointwise scalar scoring, which ArenaRL addresses through relative ranking and pairwise evaluation mechanisms.

### 7. User-Oriented Multi-Turn Dialogue Generation with Tool Use at scale

[Read Paper](https://huggingface.co/papers/2601.08225)

Large reasoning models enable scalable multi-turn dialogue generation through automated task-oriented simulation and user-oriented behavioral modeling for enhanced human-agent interaction datasets.

### 8. ShowUI-π: Flow-based Generative Models as GUI Dexterous Hands

[Read Paper](https://huggingface.co/papers/2512.24965)

Building intelligent agents capable of dexterous manipulation is essential for achieving human-like automation in both robotics and digital environments. However, existing GUI agents rely on discrete click predictions (x,y), which prohibits free-form, closed-loop trajectories (e.g. dragging a progress bar) that require continuous, on-the-fly perception and adjustment. In this work, we develop ShowUI-π, the first flow-based generative model as GUI dexterous hand, featuring the following designs: (i) Unified Discrete-Continuous Actions, integrating discrete clicks and continuous drags within a shared model, enabling flexible adaptation across diverse interaction modes; (ii) Flow-based Action Generation for drag modeling, which predicts incremental cursor adjustments from continuous visual observations via a lightweight action expert, ensuring smooth and stable trajectories; (iii) Drag Training data and Benchmark, where we manually collect and synthesize 20K drag trajectories across five domains (e.g. PowerPoint, Adobe Premiere Pro), and introduce ScreenDrag, a benchmark with comprehensive online and offline evaluation protocols for assessing GUI agents' drag capabilities. Our experiments show that proprietary GUI agents still struggle on ScreenDrag (e.g. Operator scores 13.27, and the best Gemini-2.5-CUA reaches 22.18). In contrast, ShowUI-π achieves 26.98 with only 450M parameters, underscoring both the difficulty of the task and the effectiveness of our approach. We hope this work advances GUI agents toward human-like dexterous control in digital world. The code is available at https://github.com/showlab/showui-pi.

### 9. MemoBrain: Executive Memory as an Agentic Brain for Reasoning

[Read Paper](https://huggingface.co/papers/2601.08079)

Memory management in tool-augmented agents is crucial for maintaining coherent, goal-directed reasoning over extended tasks, requiring explicit mechanisms to track and organize reasoning steps within constrained contexts.

### 10. 3AM: Segment Anything with Geometric Consistency in Videos

[Read Paper](https://huggingface.co/papers/2601.08831)

3AM enhances video object segmentation by integrating 3D-aware features from MUSt3R into SAM2, achieving improved viewpoint consistency with only RGB input at inference.
