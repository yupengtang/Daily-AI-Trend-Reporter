---
layout: post
title: "Daily AI Research Papers - Monday, May 04, 2026"
date: 2026-05-04
---

Keywords: 3D editing, 3D generative model, 3D world generation, Adjoint Matching, Distributional Implicit Value Learning, FID score, GFN, Generative Flow Networks

---

### 1. UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors

[Read Paper](https://huggingface.co/papers/2605.00658)

UniVidX is a unified multimodal framework that uses video diffusion model priors for versatile video generation through stochastic condition masking, decoupled gated LoRA, and cross-modal self-attention mechanisms.

### 2. Web2BigTable: A Bi-Level Multi-Agent LLM System for Internet-Scale Information Search and Extraction

[Read Paper](https://huggingface.co/papers/2604.27221)

Web2BigTable is a multi-agent framework that addresses both broad and deep web search challenges through a bi-level architecture with coordinated agents and iterative improvement mechanisms.

### 3. Let ViT Speak: Generative Language-Image Pre-training

[Read Paper](https://huggingface.co/papers/2605.00809)

GenLIP is a minimalist generative pretraining framework for Vision Transformers that directly predicts language tokens from visual tokens using language modeling, offering simplicity, scalability, and competitive performance in multimodal tasks.

### 4. Map2World: Segment Map Conditioned Text to 3D World Generation

[Read Paper](https://huggingface.co/papers/2605.00781)

Map2World enables 3D world generation from user-defined segment maps with improved scale consistency and detail enhancement through a pipeline leveraging asset generator priors.

### 5. From Skill Text to Skill Structure: The Scheduling-Structural-Logical Representation for Agent Skills

[Read Paper](https://huggingface.co/papers/2604.24026)

Structured representation of agent skills disentangles scheduling, execution, and logic components, improving performance in skill discovery and risk assessment tasks.

### 6. Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

[Read Paper](https://huggingface.co/papers/2605.00553)

Stable-GFN addresses training instability and mode collapse in generative flow networks for large language model red-teaming through partition function elimination and robust masking techniques.

### 7. Prox-E: Fine-Grained 3D Shape Editing via Primitive-Based Abstractions

[Read Paper](https://huggingface.co/papers/2604.23774)

A training-free framework for fine-grained 3D editing that uses geometric primitives and vision-language models to preserve identity while enabling localized structural changes.

### 8. Odysseus: Scaling VLMs to 100+ Turn Decision-Making in Games via Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2605.00347)

Given the rapidly growing capabilities of vision-language models (VLMs), extending them to interactive decision-making tasks such as video games has emerged as a promising frontier. However, existing approaches either rely on large-scale supervised fine-tuning (SFT) on human trajectories or apply reinforcement learning (RL) only in relatively short-horizon settings (typically around 20--30 turns). In this work, we study RL-based training of VLMs for long-horizon decision-making in Super Mario Land, a visually grounded environment requiring 100+ turns of interaction with coordinated perception, reasoning, and action. We begin with a systematic investigation of key algorithmic components and propose an adapted variant of PPO with a lightweight turn-level critic, which substantially improves training stability and sample efficiency over critic-free methods such as GRPO and Reinforce++. We further show that pretrained VLMs provide strong action priors, significantly improving sample efficiency during RL training and reducing the need for manual design choices such as action engineering, compared to classical deep RL trained from scratch. Building on these insights, we introduce Odysseus, an open training framework for VLM agents, achieving substantial gains across multiple levels of the game and at least 3 times average game progresses than frontier models. Moreover, the trained models exhibit consistent improvements under both in-game and cross-game generalization settings, while maintaining general-domain capabilities. Overall, our results identify key ingredients for making RL stable and effective in long-horizon, multi-modal settings, and provide practical guidance for developing VLMs as embodied agents.

### 9. Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies

[Read Paper](https://huggingface.co/papers/2605.00416)

Learning While Deploying framework enables continuous improvement of Vision-Language-Action policies through fleet-scale offline-to-online reinforcement learning with distributed robot experience and human interventions.

### 10. End-to-End Autoregressive Image Generation with 1D Semantic Tokenizer

[Read Paper](https://huggingface.co/papers/2605.00503)

End-to-end training of autoregressive image models with joint reconstruction and generation optimization achieves state-of-the-art results on ImageNet 256x256 generation.
