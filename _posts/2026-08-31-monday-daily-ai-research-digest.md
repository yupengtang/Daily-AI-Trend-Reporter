---
layout: post
title: "Daily AI Research Papers - Monday, August 31, 2026"
date: 2026-08-31
---

Keywords: Loop Engineering, coding agents, LoopArena, Controller, Worker, Loop Contract, Strict Success Rate, inference cost

---

### 1. LoopArena: Benchmarking Models as Runtime Controllers for Loop Engineering

[Read Paper](https://huggingface.co/papers/2608.28281)

LoopArena benchmarks how well a controller model guides a separate coding agent through long tasks, revealing low strict success rates and significant cost reductions.

### 2. DART-SD: Diamond-topology Aware Retrieval and Tuning for Self-Distillation of Multi-Turn Tool-Calling Agents

[Read Paper](https://huggingface.co/papers/2608.18524)

DART-SD improves multi-turn tool-calling agents by modeling execution as a diamond-topology graph, identifying critical failure points, and applying localized self-distillation to preserve valid reasoning while correcting errors.

### 3. Beyond Data Scaling: Representation-Centric Continued Pre-training for Vision-Language-Action Models

[Read Paper](https://huggingface.co/papers/2608.27550)

VLAct improves vision-language-action model performance by pre-training on diverse robot data with preserved vision-language priors and shared action semantics, achieving strong results across simulations and unseen embodiments with limited compute.

### 4. Agentic Artifact Creation: Systems, Evaluation, Principles, and Opportunities

[Read Paper](https://huggingface.co/papers/2608.28122)

This survey defines agentic artifact creation as stateful, feedback-driven construction of deliverables by AI systems and analyzes 259 works across artifact families, settings, and evaluation practices to propose principles for accountable control.

### 5. Code as Worlds: Agentic Discovery of Executable World Representations for Physical Reasoning

[Read Paper](https://huggingface.co/papers/2608.27549)

Code-as-World represents physical environments as executable code to enable quantitative reasoning and scalable supervision for vision-language models.

### 6. J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

[Read Paper](https://huggingface.co/papers/2608.26582)

J-Zero enables self-improving language models across verifiable and unverifiable domains through adversarial co-evolution of a task generator, solver, and judge using predefined preference pairs.

### 7. Revisiting Local Context for Long-Horizon Streaming 3D Reconstruction

[Read Paper](https://huggingface.co/papers/2608.27529)

ABot-Recon achieves stable long-horizon streaming 3D reconstruction by using only local temporal context and frame-independent predictions composed sequentially, reducing drift via a lightweight temporal refiner and composition-aware pose loss.

### 8. LayerRecall: A State-Conditioned Memory Router for Long-Horizon Consistency in Video Generation

[Read Paper](https://huggingface.co/papers/2608.28460)

LayerRecall selectively routes long-range historical memory into specific video diffusion layers to improve long-video consistency, supervised by cross-horizon prediction matching.

### 9. ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL

[Read Paper](https://huggingface.co/papers/2608.28476)

ContextPilot improves long-horizon agent reasoning by expanding context-editing tools and using reinforcement learning with branch sampling to identify critical context decisions.

### 10. Act with Intent: Distilling Behavior Intent for Vision-Language-Action Models

[Read Paper](https://huggingface.co/papers/2608.23478)

Vision-Language-Action (VLA) models can turn multimodal context into robot actions, but their action decoders are still trained largely by behavior cloning. This supervises which motor command was demonstrated while leaving implicit the local objective served by the behavior under the instruction. Future-based supervision enriches action learning with frames, latent observations, trajectories, or motion representations, but these signals capture particular realizations of what may happen rather than the shared semantic objective of the forthcoming behavior. We propose Intention Distillation (INDI), which distills behavior-level intent into the action decoder. During training, a frozen teacher VLM interprets a demonstrated segment from the current observation, instruction, coarse action summary, and corresponding execution video. From its standard inputs, the deployed VLA recovers the resulting multimodal intent representation at an intermediate decoder layer and uses it to organize action prediction together with representations of how the behavior unfolds and what it achieves. On SimplerEnv-Bridge, INDI improves GR00T-N1.7 from 64.3% to 84.7%, and on RoboCasa Kitchen it improves the controlled GR00T-N1.7 baseline from 64.1% to 70.3%, with consistent gains on π_{0.5} across both benchmarks. In real-world tasks, INDI improves average success from 62.0% to 68.7%, with gains of up to 12.0 pp on longer-horizon tasks. Further analyses show that the recovered latent is used by the decoder, captures behavior objective and execution progress, and organizes downstream predictions in an objective-dependent manner. These results show that action decoders benefit from explicitly modeling the semantic objective of the behavior they generate.
