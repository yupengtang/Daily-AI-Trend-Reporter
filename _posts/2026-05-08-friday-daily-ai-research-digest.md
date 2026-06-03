---
layout: post
title: "Daily AI Research Papers - Friday, May 08, 2026"
date: 2026-05-08
---

Keywords: ALFWorld, BEIR datasets, BrowseComp-Plus, EMA smoothing, GPT-4o-mini, Group Relative Policy Optimization, IR benchmarks, LLM-based agents

---

### 1. Beyond Semantic Similarity: Rethinking Retrieval for Agentic Search via Direct Corpus Interaction

[Read Paper](https://huggingface.co/papers/2605.05242)

Direct corpus interaction enables more effective agentic search by allowing agents to query raw text directly, outperforming traditional retrieval methods in complex tasks.

### 2. Skill1: Unified Evolution of Skill-Augmented Agents via Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2605.06130)

Skill1 is a unified framework that trains a single policy to simultaneously evolve skill selection, utilization, and distillation capabilities using a shared task-outcome objective, demonstrating superior performance over existing baselines in complex task environments.

### 3. Continuous Latent Diffusion Language Model

[Read Paper](https://huggingface.co/papers/2605.06548)

Cola DLM presents a hierarchical latent diffusion language model that uses text-to-latent mapping, global semantic prior modeling, and conditional decoding to achieve efficient text generation with flexible non-autoregressive inductive bias.

### 4. MiA-Signature: Approximating Global Activation for Long-Context Understanding

[Read Paper](https://huggingface.co/papers/2605.06416)

Researchers propose a compressed representation method for global activation patterns in large language models that approximates full activation states while maintaining computational efficiency and improving performance in long-context tasks.

### 5. RaguTeam at SemEval-2026 Task 8: Meno and Friends in a Judge-Orchestrated LLM Ensemble for Faithful Multi-Turn Response Generation

[Read Paper](https://huggingface.co/papers/2605.04523)

A heterogeneous ensemble of seven large language models with dual prompting strategies achieved top performance in the SemEval-2026 MTRAGEval task through judge selection and demonstrated the importance of model diversity.

### 6. SkillOS: Learning Skill Curation for Self-Evolving Agents

[Read Paper](https://huggingface.co/papers/2605.06614)

SkillOS enables self-evolving LLM agents to learn complex long-term skill curation policies through reinforcement learning, improving performance across diverse tasks while generalizing across different executor architectures.

### 7. When to Trust Imagination: Adaptive Action Execution for World Action Models

[Read Paper](https://huggingface.co/papers/2605.06222)

World Action Models (WAMs) have recently emerged as a promising paradigm for robotic manipulation by jointly predicting future visual observations and future actions. However, current WAMs typically execute a fixed number of predicted actions after each model inference, leaving the robot blind to whether the imagined future remains consistent with the actual physical rollout. In this work, we formulate adaptive WAM execution as a future-reality verification problem: the robot should execute longer when the WAM-predicted future remains reliable, and replan earlier when reality deviates from imagination. To this end, we propose Future Forward Dynamics Causal Attention (FFDC), a lightweight verifier that jointly reasons over predicted future actions, predicted visual dynamics, real observations, and language instructions to estimate whether the remaining action rollout can still be trusted. FFDC enables adaptive action chunk sizes as an emergent consequence of prediction-observation consistency, preserving the efficiency of long-horizon execution while restoring responsiveness in contact-rich or difficult phases. We further introduce Mixture-of-Horizon Training to improve long-horizon trajectory coverage for adaptive execution. Experiments on the RoboTwin benchmark and in the real world demonstrate that our method achieves a strong robustness-efficiency trade-off: on RoboTwin, it reduces WAM forward passes by 69.10% and execution time by 34.02%, while improving success rate by 2.54% over the short-chunk baseline; in real-world experiments, it improves success rate by 35%.

### 8. MARBLE: Multi-Aspect Reward Balance for Diffusion RL

[Read Paper](https://huggingface.co/papers/2605.06507)

A novel gradient-space optimization framework called MARBLE addresses limitations in multi-reward reinforcement learning fine-tuning of diffusion models by maintaining independent advantage estimators and harmonizing policy gradients through quadratic programming without manual reward weighting.

### 9. Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration

[Read Paper](https://huggingface.co/papers/2605.05566)

LoPE addresses the zero-advantage problem in reinforcement learning with verifiable rewards by usingLorem Ipsum perturbations to enhance exploration in large language model training.

### 10. Audio-Visual Intelligence in Large Foundation Models

[Read Paper](https://huggingface.co/papers/2605.04045)

Audio-Visual Intelligence represents a multidisciplinary field integrating auditory and visual modalities through large foundation models, encompassing tasks from understanding and generation to interaction, with unified taxonomies and methodological foundations.
