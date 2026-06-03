---
layout: post
title: "Daily AI Research Papers - Thursday, January 01, 2026"
date: 2026-01-01
---

Keywords: 128k context window, CoLog, Commonsense-STEM-Agent Curriculum, GaMO, Hyper-Connections (HC), Interaction-based Policy Alignment (IPA), LPIPS, Manifold-Constrained Hyper-Connections (mHC)

---

### 1. mHC: Manifold-Constrained Hyper-Connections

[Read Paper](https://huggingface.co/papers/2512.24880)

Manifold-Constrained Hyper-Connections (mHC) stabilize and scale residual connection architectures by restoring identity mapping properties through manifold projection and infrastructure optimization.

### 2. Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models

[Read Paper](https://huggingface.co/papers/2512.24618)

Youtu-LLM is a lightweight language model optimized for computational efficiency and agentic intelligence through a compact architecture, STEM-focused training curriculum, and scalable mid-training strategies for planning and reasoning tasks.

### 3. Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem

[Read Paper](https://huggingface.co/papers/2512.24873)

The Agentic Learning Ecosystem (ALE) introduces a principled infrastructure for agent development, combining post-training optimization, sandbox environments, and policy alignment to enhance long-horizon training stability and performance in real-world tasks.

### 4. A unified framework for detecting point and collective anomalies in operating system logs via collaborative transformers

[Read Paper](https://huggingface.co/papers/2512.23380)

CoLog, a log anomaly detection framework, employs collaborative transformers and multi-head impressed attention with a modality adaptation layer to achieve high-precision detection of both point and collective anomalies across diverse log modalities.

### 5. GaMO: Geometry-aware Multi-view Diffusion Outpainting for Sparse-View 3D Reconstruction

[Read Paper](https://huggingface.co/papers/2512.25073)

GaMO enhances sparse-view 3D reconstruction by using geometry-aware multi-view outpainting to improve scene coverage and consistency, achieving state-of-the-art performance with reduced computational cost.

### 6. AI Meets Brain: Memory Systems from Cognitive Neuroscience to Autonomous Agents

[Read Paper](https://huggingface.co/papers/2512.23343)

Memory serves as the pivotal nexus bridging past and future, providing both humans and AI systems with invaluable concepts and experience to navigate complex tasks. Recent research on autonomous agents has increasingly focused on designing efficient memory workflows by drawing on cognitive neuroscience. However, constrained by interdisciplinary barriers, existing works struggle to assimilate the essence of human memory mechanisms. To bridge this gap, we systematically synthesizes interdisciplinary knowledge of memory, connecting insights from cognitive neuroscience with LLM-driven agents. Specifically, we first elucidate the definition and function of memory along a progressive trajectory from cognitive neuroscience through LLMs to agents. We then provide a comparative analysis of memory taxonomy, storage mechanisms, and the complete management lifecycle from both biological and artificial perspectives. Subsequently, we review the mainstream benchmarks for evaluating agent memory. Additionally, we explore memory security from dual perspectives of attack and defense. Finally, we envision future research directions, with a focus on multimodal memory systems and skill acquisition.

### 7. Pretraining Frame Preservation in Autoregressive Video Memory Compression

[Read Paper](https://huggingface.co/papers/2512.23851)

We present PFP, a neural network structure to compress long videos into short contexts, with an explicit pretraining objective to preserve the high-frequency details of single frames at arbitrary temporal positions. The baseline model can compress a 20-second video into a context at about 5k length, where random frames can be retrieved with perceptually preserved appearances. Such pretrained models can be directly fine-tuned as memory encoders for autoregressive video models, enabling long history memory with low context cost and relatively low fidelity loss. We evaluate the framework with ablative settings and discuss the trade-offs of possible neural architecture designs.

### 8. GR-Dexter Technical Report

[Read Paper](https://huggingface.co/papers/2512.24210)

GR-Dexter introduces a hardware-model-data framework for bimanual dexterous-hand robot manipulation using vision-language-action models, combining teleoperation data and multimodal datasets to achieve robust generalization.

### 9. PhyGDPO: Physics-Aware Groupwise Direct Preference Optimization for Physically Consistent Text-to-Video Generation

[Read Paper](https://huggingface.co/papers/2512.24551)

Recent advances in text-to-video (T2V) generation have achieved good visual quality, yet synthesizing videos that faithfully follow physical laws remains an open challenge. Existing methods mainly based on graphics or prompt extension struggle to generalize beyond simple simulated environments or learn implicit physical reasoning. The scarcity of training data with rich physics interactions and phenomena is also a problem. In this paper, we first introduce a Physics-Augmented video data construction Pipeline, PhyAugPipe, that leverages a vision-language model (VLM) with chain-of-thought reasoning to collect a large-scale training dataset, PhyVidGen-135K. Then we formulate a principled Physics-aware Groupwise Direct Preference Optimization, PhyGDPO, framework that builds upon the groupwise Plackett-Luce probabilistic model to capture holistic preferences beyond pairwise comparisons. In PhyGDPO, we design a Physics-Guided Rewarding (PGR) scheme that embeds VLM-based physics rewards to steer optimization toward physical consistency. We also propose a LoRA-Switch Reference (LoRA-SR) scheme that eliminates memory-heavy reference duplication for efficient training. Experiments show that our method significantly outperforms state-of-the-art open-source methods on PhyGenBench and VideoPhy2. Please check our project page at https://caiyuanhao1998.github.io/project/PhyGDPO for more video results. Our code, models, and data will be released at https://github.com/caiyuanhao1998/Open-PhyGDPO

### 10. Scaling Open-Ended Reasoning to Predict the Future

[Read Paper](https://huggingface.co/papers/2512.25070)

High-stakes decision making involves reasoning under uncertainty about the future. In this work, we train language models to make predictions on open-ended forecasting questions. To scale up training data, we synthesize novel forecasting questions from global events reported in daily news, using a fully automated, careful curation recipe. We train the Qwen3 thinking models on our dataset, OpenForesight. To prevent leakage of future information during training and evaluation, we use an offline news corpus, both for data generation and retrieval in our forecasting system. Guided by a small validation set, we show the benefits of retrieval, and an improved reward function for reinforcement learning (RL). Once we obtain our final forecasting system, we perform held-out testing between May to August 2025. Our specialized model, OpenForecaster 8B, matches much larger proprietary models, with our training improving the accuracy, calibration, and consistency of predictions. We find calibration improvements from forecasting training generalize across popular benchmarks. We open-source all our models, code, and data to make research on language model forecasting broadly accessible.
