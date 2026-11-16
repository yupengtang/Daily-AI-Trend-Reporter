---
layout: post
title: "Daily AI Research Papers - Monday, November 16, 2026"
date: 2026-11-16
---

Keywords: multimodal benchmarks, agentic robotics, foundation model, agentic harnesses, task-specific policies, embodied benchmarks, closed-loop system, Embodied System 2

---

### 1. Envisioning Agentic Robotics: A New Era of Robotic Systems

[Read Paper](https://huggingface.co/papers/2611.18359)

Agentic robotics, centered on an Embodied System 2 formed by a foundation model and embodied harness, requires a shared evaluation framework and research roadmap to achieve reliable physical autonomy.

### 2. Gamer: A Generalist Vision-Language-Action Foundation Model for Long-Horizon Real-Time Game Agent

[Read Paper](https://huggingface.co/papers/2611.18397)

Gamer is a generalist agent framework that integrates perception, memory, and multi-step reasoning, trained through continued pre-training, sparse-thinking instruction tuning, and multi-turn reinforcement learning to interact efficiently in 2D and 3D environments.

### 3. VibePrevisBench: Evaluating Agentic Coding for 4D Scene Generation in Blender

[Read Paper](https://huggingface.co/papers/2611.15789)

Controllable real-world video generation models can condition on a rendered graybox proxy video, which fixes layout, motion and camera. Coding agents can build such videos by reconstructing a 4D scene in Blender, yet no benchmark checks both these videos and the long, multi-shot scenes behind them. We introduce VibePrevisBench, a benchmark for 4D scene reconstruction: an agent turns a brief into an executable 4D scene and renders its proxy video. Its 242 tasks span four input kinds, including scripts, photos, continuations, and reference videos. Each delivery is scored along two evaluation dimensions. Rule-based Scene Check re-runs the code and compares the exported 4D scene with hidden ground truth facts in nine dimensions. VLM-based Screen Check asks a VLM judge whether the facts that pass can be recognized in the proxy video. We analyze the evaluation results and trajectories of 11 frontier coding agents and identify three key observations. First, multimodal understanding separates the agents, especially in understanding space and camera motion. Second, all agents seem to already master basic camera work and staging, and fail at challenging requests. Third, agents inspect their own renders but fail to correct their errors. Agents thus need better visual understanding of space and camera, explicit checks for challenging requests, and item-by-item verification of renders against the brief.

### 4. MIRIX-2.0: Multi-Agent Memory System

[Read Paper](https://huggingface.co/papers/2611.12068)

MIRIX-2.0 is a multi-agent memory system for LLM agents that achieves high accuracy and efficiency across QA and procedural benchmarks without fine-tuning.

### 5. MMFlash: Faster Speculative Decoding for Vision-Language Models

[Read Paper](https://huggingface.co/papers/2611.18482)

MMFlash improves multimodal speculative decoding by adaptively upweighting visually dependent tokens and using sparse attention for long video, yielding up to 3.58× lossless speedup.

### 6. Rethinking in Spikes: Mitigating Hallucinations in MDLMs with Step-Aware Decoding

[Read Paper](https://huggingface.co/papers/2611.12901)

StepRefine, a plug-and-play decoding strategy, mitigates hallucinations in multimodal diffusion language models by switching between discrete denoising and latent refinement based on entropy spikes and distributional stability.

### 7. V-JEPA-LM: Video-Language Models That Read Dynamics Beyond Static, Text-Aligned Representations

[Read Paper](https://huggingface.co/papers/2611.17831)

Counterfactual supervision over pixel transformations enables language models to read motion dynamics from frozen predictive video encoders, yielding V-JEPA-LM models that outperform open video LLMs across benchmarks.

### 8. DySENSE: Dynamic Spatiotemporal ENergy Synthesis and Emulation with Satellite Imagery

[Read Paper](https://huggingface.co/papers/2611.17878)

Cities account for over two-thirds of the world's energy and more than 70% of global carbon dioxide emissions. City energy is essential for achieving the 2050 net-zero target and the UN's Sustainable Development Goals 7 and 11. Although existing studies have made progress, they are limited by data and methodological gaps. Existing studies are largely predictive for existing cities rather than generative for future cities; fine-grained, aligned monthly dynamic energy data are scarce; and existing GenAI models struggle to generate dynamic urban functional monthly energy. We present DySENSE, a joint multimodal GenAI framework that synthesizes static urban satellite imagery together with spatially aligned, dynamic monthly building energy consumption maps. We decouple long-term static urban morphology from short-term dynamic energy consumption. By conditioning on static urban land use, DySENSE can generate realistic, diverse and new static urban imagery with aligned energy maps. By conditioning on dynamic monthly and climate prompts, DySENSE can jointly generate dynamic, monthly energy consumption maps while keeping the urban morphology static. Experiments across four cities (New York, Washington, Norfolk, Busan) demonstrated that DySENSE can generate annotated high-fidelity synthetic energy data using less than 25% labelled data, boosting downstream prediction performance by 8% mIoU, and achieving strong energy dynamic consistency. Compared to existing methods, DySENSE significantly improved prediction performance (8.7% accuracy) and reduced prediction error (9%-18% CVRMSE, ASHRAE). DySENSE offers an effective foundation for energy-efficient cities, peak-load analysis, building performance simulation and dynamic demand response for city, energy, and building science. The data and code: https://huggingface.co/datasets/skl24/MUSE2 and https://github.com/kailaisun/DySENSE

### 9. Loopback Transformer: Towards a Unified View of Residual Looped Transformers

[Read Paper](https://huggingface.co/papers/2611.07226)

A looped Transformer that separately controls input strength, total update weight, and per-update allocation achieves lower validation loss and higher downstream accuracy than standard and looped residual baselines while using fewer parameters.

### 10. G-CAST: Governed Continual Self-Training for E-Commerce Agents under Evolving Demands

[Read Paper](https://huggingface.co/papers/2611.17814)

With the rapid advancement of Large Language Models (LLMs), an increasing number of users are turning to LLM-powered agents for conversational shopping in e-commerce. However, the user needs continue to change as new product categories, procurement scenarios, and feature requests emerge. At the same time, the product's execution harness-including its tools, skills, and operating procedures-continues to evolve to support increasingly sophisticated interactions. These changes create two practical challenges: 1) Demand drift alters the distribution, intent, and complexity of user requests, causing a model trained once to fall behind; 2) Harness evolution requires the model to use newly introduced capabilities correctly without losing behavior that users already rely on. Periodic manual retraining is too delayed and labor-intensive for this setting. To address these challenges, we present G-CAST, a governed continual self-training system that converts daily production evidence under evolving demand and capability contracts into reward-routed, entropy-prioritized training data and governed reward evolution. Governed checkpoint transition under cumulative evaluation controls each model update, enabling the agent to learn newly introduced behaviors without sacrificing previously accepted capabilities. Experiments on a real production e-commerce agent show that G-CAST achieves stronger adaptation to newly introduced demand while better preserving previously accepted behavior compared with a strong autonomous post-training baseline. Ablations further show that both reward-routed, entropy-prioritized data orchestration and governed reward evolution are important to these gains. G-CAST is now deployed in Yuancai(http://cai.1688.com/), a B2B procurement agent on Alibaba's 1688 platform.
