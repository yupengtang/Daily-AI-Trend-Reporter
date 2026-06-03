---
layout: post
title: "Daily AI Research Papers - Friday, December 26, 2025"
date: 2025-12-26
---

Keywords: spatial memory-aware, video generation framework, 3D scene point cloud, dynamic-static disentanglement, visual SLAM, explicit camera control, 3D-aware interactive editing, autoregressive (AR) visual generation

---

### 1. Latent Implicit Visual Reasoning

[Read Paper](https://huggingface.co/papers/2512.21218)

While Large Multimodal Models (LMMs) have made significant progress, they remain largely text-centric, relying on language as their core reasoning modality. As a result, they are limited in their ability to handle reasoning tasks that are predominantly visual. Recent approaches have sought to address this by supervising intermediate visual steps with helper images, depth maps, or image crops. However, these strategies impose restrictive priors on what "useful" visual abstractions look like, add heavy annotation costs, and struggle to generalize across tasks. To address this critical limitation, we propose a task-agnostic mechanism that trains LMMs to discover and use visual reasoning tokens without explicit supervision. These tokens attend globally and re-encode the image in a task-adaptive way, enabling the model to extract relevant visual information without hand-crafted supervision. Our approach outperforms direct fine-tuning and achieves state-of-the-art results on a diverse range of vision-centric tasks -- including those where intermediate abstractions are hard to specify -- while also generalizing to multi-task instruction tuning.

### 2. Emergent temporal abstractions in autoregressive models enable hierarchical reinforcement learning

[Read Paper](https://huggingface.co/papers/2512.20605)

Large-scale autoregressive models pretrained on next-token prediction and finetuned with reinforcement learning (RL) have achieved unprecedented success on many problem domains. During RL, these models explore by generating new outputs, one token at a time. However, sampling actions token-by-token can result in highly inefficient learning, particularly when rewards are sparse. Here, we show that it is possible to overcome this problem by acting and exploring within the internal representations of an autoregressive model. Specifically, to discover temporally-abstract actions, we introduce a higher-order, non-causal sequence model whose outputs control the residual stream activations of a base autoregressive model. On grid world and MuJoCo-based tasks with hierarchical structure, we find that the higher-order model learns to compress long activation sequence chunks onto internal controllers. Critically, each controller executes a sequence of behaviorally meaningful actions that unfold over long timescales and are accompanied with a learned termination condition, such that composing multiple controllers over time leads to efficient exploration on novel tasks. We show that direct internal controller reinforcement, a process we term "internal RL", enables learning from sparse rewards in cases where standard RL finetuning fails. Our results demonstrate the benefits of latent action generation and reinforcement in autoregressive models, suggesting internal RL as a promising avenue for realizing hierarchical RL within foundation models.

### 3. Spatia: Video Generation with Updatable Spatial Memory

[Read Paper](https://huggingface.co/papers/2512.15716)

Spatia, a spatial memory-aware video generation framework, maintains long-term spatial and temporal consistency by preserving and updating a 3D scene point cloud, enabling realistic video generation and interactive editing.

### 4. Schoenfeld's Anatomy of Mathematical Reasoning by Language Models

[Read Paper](https://huggingface.co/papers/2512.19995)

Large language models increasingly expose reasoning traces, yet their underlying cognitive structure and steps remain difficult to identify and analyze beyond surface-level statistics. We adopt Schoenfeld's Episode Theory as an inductive, intermediate-scale lens and introduce ThinkARM (Anatomy of Reasoning in Models), a scalable framework that explicitly abstracts reasoning traces into functional reasoning steps such as Analysis, Explore, Implement, Verify, etc. When applied to mathematical problem solving by diverse models, this abstraction reveals reproducible thinking dynamics and structural differences between reasoning and non-reasoning models, which are not apparent from token-level views. We further present two diagnostic case studies showing that exploration functions as a critical branching step associated with correctness, and that efficiency-oriented methods selectively suppress evaluative feedback steps rather than uniformly shortening responses. Together, our results demonstrate that episode-level representations make reasoning steps explicit, enabling systematic analysis of how reasoning is structured, stabilized, and altered in modern language models.

### 5. VA-π: Variational Policy Alignment for Pixel-Aware Autoregressive Generation

[Read Paper](https://huggingface.co/papers/2512.19680)

VA-$\pi$ optimizes autoregressive visual generators using a pixel-space objective to improve image quality and performance without retraining tokenizers or using external rewards.

### 6. How Much 3D Do Video Foundation Models Encode?

[Read Paper](https://huggingface.co/papers/2512.19949)

Videos are continuous 2D projections of 3D worlds. After training on large video data, will global 3D understanding naturally emerge? We study this by quantifying the 3D understanding of existing Video Foundation Models (VidFMs) pretrained on vast video data. We propose the first model-agnostic framework that measures the 3D awareness of various VidFMs by estimating multiple 3D properties from their features via shallow read-outs. Our study presents meaningful findings regarding the 3D awareness of VidFMs on multiple axes. In particular, we show that state-of-the-art video generation models exhibit a strong understanding of 3D objects and scenes, despite not being trained on any 3D data. Such understanding can even surpass that of large expert models specifically trained for 3D tasks. Our findings, together with the 3D benchmarking of major VidFMs, provide valuable observations for building scalable 3D models.

### 7. GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training

[Read Paper](https://huggingface.co/papers/2512.13043)

Multi-turn reinforcement learning (RL) for multi-modal agents built upon vision-language models (VLMs) is hampered by sparse rewards and long-horizon credit assignment. Recent methods densify the reward by querying a teacher that provides step-level feedback, e.g., Guided Thought Reinforcement (GTR) and On-Policy Distillation, but rely on costly, often privileged models as the teacher, limiting practicality and reproducibility. We introduce GTR-Turbo, a highly efficient upgrade to GTR, which matches the performance without training or querying an expensive teacher model. Specifically, GTR-Turbo merges the weights of checkpoints produced during the ongoing RL training, and then uses this merged model as a "free" teacher to guide the subsequent RL via supervised fine-tuning or soft logit distillation. This design removes dependence on privileged VLMs (e.g., GPT or Gemini), mitigates the "entropy collapse" observed in prior work, and keeps training stable. Across diverse visual agentic tasks, GTR-Turbo improves the accuracy of the baseline model by 10-30% while reducing wall-clock training time by 50% and compute cost by 60% relative to GTR.
