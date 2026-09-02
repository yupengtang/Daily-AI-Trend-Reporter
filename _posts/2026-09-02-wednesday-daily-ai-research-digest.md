---
layout: post
title: "Daily AI Research Papers - Wednesday, September 02, 2026"
date: 2026-09-02
---

Keywords: student simulators, pooled training, per-student specialization, behavioral fidelity, guidance responsiveness, reinforcement learning, reward model, vision-language foundation model

---

### 1. StudentSim: Training LLM-based Student Simulators

[Read Paper](https://huggingface.co/papers/2609.01591)

StudentSim trains personalized student simulators from sparse data to mirror learner responses and adapt to tutor guidance, outperforming existing models across chess, writing, and math.

### 2. Qwen-Drive-1.0: An Initial Step towards a Vision-Language Foundation Model for Autonomous Driving

[Read Paper](https://huggingface.co/papers/2609.00111)

Qwen-Drive-1.0 is a vision-language foundation model for autonomous driving that unifies 3D perception, visual question answering, and motion planning via shared representations and staged training.

### 3. SMELT: Scaling Laws for Compute-Matched MoE Looped Transformers

[Read Paper](https://huggingface.co/papers/2609.01343)

Looping middle layers in sparse Mixture-of-Experts Transformers improves training efficiency and downstream performance while matching per-token FLOPs, parameters, and cache budgets.

### 4. UI-Venus-2 Technical Report

[Read Paper](https://huggingface.co/papers/2609.00028)

UI-Venus-2 is a general-purpose multimodal GUI agent that uses unified reasoning-action loops, expanded environment coverage, and robust verification to enable reliable real-world digital automation.

### 5. H3-World: Turning Language Understanding into World Control

[Read Paper](https://huggingface.co/papers/2609.01560)

We present H3-World, an efficient framework that turns the 33B MiniMax-H3 video generator into an interactive world model. Our key finding is that, as large video generators become more capable, language is emerging as a natural interface for control. MiniMax-H3, for example, already supports zero-shot control of character behavior and camera motion through natural-language instructions. Building on this, H3-World turns this coarse language interface into precise, temporally grounded world control, without introducing dedicated action modules. Specifically, we represent each action as a structured combination of character and camera instructions, and align them with the corresponding temporal video latents. To make the control temporally precise, we further introduce temporal attention routing, which restricts each instruction to its intended time interval and reduces control leakage across actions. Importantly, H3-World directly reuses the semantic representations learned during large-scale video pretraining and requires only lightweight adaptation. With only 8,000 gameplay samples, 10,000 LoRA optimization steps, and 0.199% trainable parameters, H3-World achieves effective character and camera control while preserving strong generation quality. It also generalizes to unseen scenarios. These results show that the control capabilities emerging in large video generators can be efficiently transformed into interactive world control.

### 6. ZimaBlue: Evolving Generalizable World Action Models through Scalable Video Pre-training

[Read Paper](https://huggingface.co/papers/2609.00188)

ZimaBlue learns generalizable world action models from large-scale egocentric video via a three-stage curriculum and a slow-fast architecture, substantially improving zero-shot robotic manipulation.

### 7. From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix

[Read Paper](https://huggingface.co/papers/2609.01572)

A smaller self-hosted LLM trained with separate GRPO experts merged via SLERP outperforms a much larger baseline on instruction following, function-calling, and internal tasks while serving half of platform traffic at lower cost.

### 8. Hi-Q: Hierarchical Evidence-guided Query Refinement for Multi-Hop Question Answering

[Read Paper](https://huggingface.co/papers/2608.30468)

Hi-Q is an evidence-conditioned framework that dynamically refines multi-hop queries into hierarchical trees guided by corpus support signals, improving retrieval and answer accuracy without fixed graphs.

### 9. Evaluating Multimodal LLMs as Generalist Vision-Language-Action Agents for Drone Control: Commanding, Approaching, Tracking and Searching

[Read Paper](https://huggingface.co/papers/2609.01404)

DroneCATS evaluates multimodal language models as drone controllers and finds that small open models navigate well but fail at protocol adherence and termination, highlighting a gap between spatial perception and disciplined action.

### 10. Uncovering Understanding-Generation Synergy in Native Unified Multimodal Models: From Representation, Task to System

[Read Paper](https://huggingface.co/papers/2609.01607)

Unified multimodal models achieve synergy between visual understanding and generation through specialized architectures, shared knowledge, and end-to-end optimization rather than simple functional unification.
