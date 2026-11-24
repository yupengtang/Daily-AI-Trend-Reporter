---
layout: post
title: "Daily AI Research Papers - Tuesday, November 24, 2026"
date: 2026-11-24
---

Keywords: planning, vision-language models, reinforcement learning, frontier multimodal models, robot policies, closed-loop robotic harnesses, frontier-model-as-policy, prompting

---

### 1. Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control

[Read Paper](https://huggingface.co/papers/2611.29897)

Long-horizon, real-time controllable motion generation is essential for interactive animation and humanoid robotics, where an agent must respond to changing instructions and spatial goals while moving. Existing autoregressive diffusion methods generate motion incrementally, but bounded attention caches limit access to distant history, while repeated denoising introduces substantial inference overhead. We present Motion Mamba 2.0, a Mamba-based root-body diffusion architecture that integrates local attention, resumable recurrent states, and persistent content memory (PCM) for long-horizon controllable motion generation. PCM transforms compressed motion history from Mamba recurrent states, together with state changes and text-conditioned features, into dynamically updated content slots accessible to local attention. To enable efficient inference, we further develop an execution scheme tailored to this architecture, combining CUDA Graph replay, fused kernels, and execution-flow simplification while preserving the canonical history update. Evaluations on retargeted HiPHI demonstrate accurate terminal control over rollouts of up to 39.52 s, while short-horizon comparisons on BONES Seed and retargeted HumanML3D show that the model also retains strong spatial control and text-motion matching. On an RTX 4090, the optimized execution reduces median complete-window latency from 1685.35 to 276.14 ms, achieving a median paired speedup of 6.138times while preserving identical outputs and committed states in the tested rollouts. Moreover, we conduct extensive real-world evaluation on the Unitree G1, demonstrating sequential text-guided action composition, expressive motion, and interactive locomotion through keyboard and mouse control.

### 2. Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness

[Read Paper](https://huggingface.co/papers/2611.31201)

A systematic framework shows that frontier multimodal models can serve as robot policies when harnessed with interfaces favoring native visual reasoning and tool use, achieving high success in simulated and real manipulation tasks.

### 3. WorldLeague: Benchmarking Physical Agency

[Read Paper](https://huggingface.co/papers/2611.30989)

WorldLeague benchmarks 13 AI systems on 60 simulated robot tasks spanning exploration, rearrangement, and adaptation, measuring progress and success to compare autonomous physical goal achievement.

### 4. 3AGameFactory-Motion: Agentic Rigging and Reliable Complex Motion Generation for Diverse 3D Assets

[Read Paper](https://huggingface.co/papers/2611.30013)

A new benchmark evaluates rigging and motion generation across six 3D morphology categories, compares specialized, video-based, and agent-driven paradigms, and proposes an agentic framework that writes compact parametric skeleton and motion programs executed deterministically for reliable control.

### 5. A Glimpse of Training VLMs as Foundation Models for Robotic Manipulation

[Read Paper](https://huggingface.co/papers/2611.29338)

nanoAstra combines semantic action modeling with capability-preserving supervised fine-tuning to train vision-language models for generalizable robot manipulation, achieving strong zero-shot success and faster streaming execution.

### 6. WeMM-Ecom: Advancing Multimodal Content Understanding for WeChat E-Commerce

[Read Paper](https://huggingface.co/papers/2611.30662)

WeMM-Ecom, a family of multimodal large language models spanning dense and mixture-of-experts sizes, is trained with continued pre-training and hybrid-thinking post-training to achieve strong e-commerce and general-purpose results and is deployed in WeChat.

### 7. Let the Fixer Decide: Training a Code Localizer from the Fixer's Likelihood

[Read Paper](https://huggingface.co/papers/2611.12164)

A code localizer is trained with scores from a frozen fixer via weighted supervised fine-tuning, improving bug localization and patch success for a small coding agent without tests or reinforcement learning.

### 8. ViDecide: Teaching Video Diffusion Models to Reason and Plan Step by Step

[Read Paper](https://huggingface.co/papers/2611.26389)

ViDecide trains video diffusion models to make causal state-transition and stopping decisions, using Corrective Fine-tuning to improve multi-step planning success without policy gradients.

### 9. AutoPILOT: A Visual Agent Harness for Real-Time Gameplay

[Read Paper](https://huggingface.co/papers/2611.28202)

A harness that lets multimodal foundation models emit primitive actions, open-loop macros, or closed-loop controllers improves real-time game performance and reduces API cost.

### 10. Plan4D: Generative Plannable 4D Worlds

[Read Paper](https://huggingface.co/papers/2611.26516)

Most world models for future prediction prioritize aestetic quality of generated videos, but offer only weak physical grounding and limited geometric consistency over time, due to the lack of explicit 4D modeling. In this paper, we introduce plan4d, a possible next milestone towards building a generative plannable 4D world model that predicts future RGB videos together with dense 4D geometry, including depth, cameras, point clouds, and scene flow from an observed scene context and text-specified actions. Since training data with paired actions and dense 4D annotations is inherently scarce, plan4d leverages the complementary priors of foundational video generators and 3D reconstructors. Specifically, we propose a model-stitching architecture that couples Depth Anything 3 with LongCat-Video and augments it with an auxiliary DPT head for scene-flow prediction. The resulting model allows us to plan and generate complex, long-horizon, and fine-grained actions in a dynamic 3D scenes, while substantially improving both visual fidelity and geometric accuracy of the generated future videos. Extensive experiments on both synthetic and real-world datasets demonstrate that plan4d achieves state-of-the-art performance in generative 4D modeling of environmental reactions. Project Page: https://fogradio.github.io/plan4d-project
