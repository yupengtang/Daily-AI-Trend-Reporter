---
layout: post
title: "Daily AI Research Papers - Thursday, November 05, 2026"
date: 2026-11-05
---

Keywords: Group Relative Policy Optimization, turn-level credit assignment, silence-completed prefix scoring, turn-mean policy optimization, streaming video understanding, vision-language-action models, visual encoders, flow-matching

---

### 1. Turns Matter: Advancing Streaming Video RL via Turn-Level Credit

[Read Paper](https://huggingface.co/papers/2611.06728)

TurnWise improves streaming video understanding by assigning turn-level credit and balancing decision weights in group relative policy optimization.

### 2. From Pixels to Actions - Towards Native Vision-Language-Action Policies

[Read Paper](https://huggingface.co/papers/2611.06918)

NEO-VLA eliminates fixed visual encoders to let action policies access evolving visual representations across network depths, improving manipulation robustness.

### 3. FrontierPhysics: Evaluating Agents for End-to-End Physics Research

[Read Paper](https://huggingface.co/papers/2611.06910)

FrontierPhysics is an end-to-end benchmark of 56 expert-designed physics tasks that evaluates AI agents on literature review, research planning, experimentation, and manuscript production using deterministic verifiers and expert rubrics.

### 4. GameVerse: A Minute-Scale Gameplay Dataset for Long-Horizon Interactive World Modeling

[Read Paper](https://huggingface.co/papers/2610.37257)

GameVerse is a large-scale gameplay dataset with minute-long sequences and event-aligned annotations that improves long-horizon interactive world modeling.

### 5. What Does Multi-Turn Reinforcement Learning Really Improve in LLM Agents?

[Read Paper](https://huggingface.co/papers/2611.06483)

While multi-turn agentic reinforcement learning (RL) excels on long-horizon tasks, its performance gains remain highly sensitive to interaction constraints. Evaluating across sampling breadth K and interaction depth T via Pass@(K,T), we show that trajectory-level RL advantages depend on sufficient turn budgets, diminishing significantly when budgets tighten. Besides, agentic RL does not uniformly elevate single-step decision quality; rather, its strength concentrates in error recovery and long-horizon resilience. To mitigate the deployment risks caused by this budget dependence, we provide a preliminary discussion on the benefits and limitations of process supervision signals and introduce BudgetMix RL as an alternative that varies interaction budgets during training. Experiments demonstrate that BudgetMix RL achieves robust performance across diverse execution settings. Project Page: https://rethinking-agentic-rl.github.io/

### 6. Halley: A Predictive Control Plane for Optical AI Data Center Networks

[Read Paper](https://huggingface.co/papers/2611.04408)

Halley is a predictive control framework for AI data center optical networks that improves effective throughput and reduces capacity needs by forecasting failures and controlling resources with formal guarantees.

### 7. EmoReward: Self-Evolving Reward Calibration for Emotional Video Generation RL

[Read Paper](https://huggingface.co/papers/2611.06518)

Group Relative Policy Optimization (GRPO) has advanced generative model post-training in verifiable domains such as code and mathematics, but evaluating video quality, especially emotional expression, often depends on subjective aesthetics and has no uniquely correct target. Existing approaches typically rely on fixed rewards for visual quality, motion quality, and text alignment that poorly capture emotional intent and can invite reward hacking, while RLHF is difficult to scale and subject to individual aesthetic differences that complicate the calibration of facial, body motion, and chromatic cues across narrative contexts. We introduce EmoReward, a new self-evolving training mechanism that distills evidence reliability from reward traces collected during an actual training run to adaptively guide multi-reward weight allocation in subsequent training by determining which emotional evidence to trust and how much to weight it. EmoReward first plans specific grounding cues from the emotional prompt to guide evaluation, then uses heterogeneous signal primitives to score the generated video during the act stage. Reflective Reward Gating calibrates their contributions through instance-level gates for phase, face confidence, conflict, and subtlety, then renormalizes weights over the active set to produce the sample's fused reward. RewardMemory then estimates each primitive's reliability from rollout scores and outcomes within each emotional regime in the finished training run, then fixes bounded multipliers that calibrate fusion weights in the next training run without updating reward models. Experiments on HunyuanVideo 1.5 show improvements across multiple RL algorithms and compare EmoReward with other agentic reward methods under SAGE-GRPO. Ablations and transfer experiments on SD 3.5 and FLUX further support adaptive reward calibration for visual generation.

### 8. PyRUA-DataFarm: LLM Agent as a Cost-Efficient Teleoperator for Real-Ready Robot Data

[Read Paper](https://huggingface.co/papers/2611.05320)

A coding agent automates robot demonstration generation by simulating tasks, rendering photorealistic videos, and producing deployable training data at lower cost than teleoperation.

### 9. StreamDuo: A Live Commentator and a Research Agent for Unseen Games

[Read Paper](https://huggingface.co/papers/2611.01120)

StreamDuo pairs a trainable commentator with a frozen research agent that retrieves game wiki entries to improve cross-game streaming commentary without training on target games.

### 10. Solv3R: 3D Looped Transformer as a Multi-view Solver

[Read Paper](https://huggingface.co/papers/2611.02857)

Solv3R is a looped transformer that iteratively refines multi-view geometry from its current state, supporting incremental updates, caching, and multiple reconstruction tasks without specialized modules.
