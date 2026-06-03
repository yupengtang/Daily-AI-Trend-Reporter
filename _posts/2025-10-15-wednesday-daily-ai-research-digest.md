---
layout: post
title: "Daily AI Research Papers - Wednesday, October 15, 2025"
date: 2025-10-15
---

Keywords: Reinforcement Learning, Behavioral Cloning, language-conditioned models, large language models, machine translation, web novels, DITING, narrative fidelity

---

### 1. Spatial Forcing: Implicit Spatial Representation Alignment for
  Vision-language-action Model

[Read Paper](https://huggingface.co/papers/2510.12276)

Vision-language-action (VLA) models have recently shown strong potential in
enabling robots to follow language instructions and execute precise actions.
However, most VLAs are built upon vision-language models pretrained solely on
2D data, which lack accurate spatial awareness and hinder their ability to
operate in the 3D physical world. Existing solutions attempt to incorporate
explicit 3D sensor inputs such as depth maps or point clouds, but these
approaches face challenges due to sensor noise, hardware heterogeneity, and
incomplete depth coverage in existing datasets. Alternative methods that
estimate 3D cues from 2D images also suffer from the limited performance of
depth estimators.We propose Spatial Forcing (SF), a simple yet effective
alignment strategy that implicitly forces VLA models to develop spatial
comprehension capabilities without relying on explicit 3D inputs or depth
estimators. SF aligns intermediate visual embeddings of VLAs with geometric
representations produced by pretrained 3D foundation models. By enforcing
alignment at intermediate layers, SF guides VLAs to encode richer spatial
representations that enhance action precision.Extensive experiments in
simulation and real-world environments demonstrate that SF achieves
state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further
accelerates training by up to 3.8x and improves data efficiency across diverse
robotic tasks. Project page is at https://spatial-forcing.github.io/

### 2. Robot Learning: A Tutorial

[Read Paper](https://huggingface.co/papers/2510.12403)

Robot learning transitions from model-based to data-driven methods, leveraging reinforcement learning and behavioral cloning to develop versatile, language-conditioned models for diverse tasks and robot types.

### 3. Advancing End-to-End Pixel Space Generative Modeling via Self-supervised
  Pre-training

[Read Paper](https://huggingface.co/papers/2510.12586)

Pixel-space generative models are often more difficult to train and generally
underperform compared to their latent-space counterparts, leaving a persistent
performance and efficiency gap. In this paper, we introduce a novel two-stage
training framework that closes this gap for pixel-space diffusion and
consistency models. In the first stage, we pre-train encoders to capture
meaningful semantics from clean images while aligning them with points along
the same deterministic sampling trajectory, which evolves points from the prior
to the data distribution. In the second stage, we integrate the encoder with a
randomly initialized decoder and fine-tune the complete model end-to-end for
both diffusion and consistency models. Our training framework demonstrates
strong empirical performance on ImageNet dataset. Specifically, our diffusion
model reaches an FID of 2.04 on ImageNet-256 and 2.35 on ImageNet-512 with 75
number of function evaluations (NFE), surpassing prior pixel-space methods by a
large margin in both generation quality and efficiency while rivaling leading
VAE-based models at comparable training cost. Furthermore, on ImageNet-256, our
consistency model achieves an impressive FID of 8.82 in a single sampling step,
significantly surpassing its latent-space counterpart. To the best of our
knowledge, this marks the first successful training of a consistency model
directly on high-resolution images without relying on pre-trained VAEs or
diffusion models.

### 4. Scaling Language-Centric Omnimodal Representation Learning

[Read Paper](https://huggingface.co/papers/2510.11693)

Recent multimodal embedding approaches leveraging multimodal large language
models (MLLMs) fine-tuned with contrastive learning (CL) have shown promising
results, yet the underlying reasons behind their superiority remain
underexplored. This work argues that a crucial advantage of MLLM-based
approaches stems from implicit cross-modal alignment achieved during generative
pretraining, where the language decoder learns to exploit multimodal signals
within a shared representation space for generating unimodal outputs. Through
analysis of anisotropy and kernel similarity structure, we empirically confirm
that latent alignment emerges within MLLM representations, allowing CL to serve
as a lightweight refinement stage. Leveraging this insight, we propose a
Language-Centric Omnimodal Embedding framework, termed LCO-Emb. Extensive
experiments across diverse backbones and benchmarks demonstrate its
effectiveness, achieving state-of-the-art performance across modalities.
Furthermore, we identify a Generation-Representation Scaling Law (GRSL),
showing that the representational capabilities gained through contrastive
refinement scales positively with the MLLM's generative capabilities. This
suggests that improving generative abilities evolves as an effective paradigm
for enhancing representation quality. We provide a theoretical explanation of
GRSL, which formally links the MLLM's generative quality to the upper bound on
its representation performance, and validate it on a challenging, low-resource
visual-document retrieval task, showing that continual generative pretraining
before CL can further enhance the potential of a model's embedding
capabilities. Codes, models, and resources are available at
https://github.com/LCO-Embedding/LCO-Embedding.

### 5. DITING: A Multi-Agent Evaluation Framework for Benchmarking Web Novel
  Translation

[Read Paper](https://huggingface.co/papers/2510.09116)

A new evaluation framework, DITING, and a reasoning-driven multi-agent evaluation framework, AgentEval, are introduced to assess the quality of web novel translations, revealing that Chinese-trained LLMs outperform larger foreign models.

### 6. RAG-Anything: All-in-One RAG Framework

[Read Paper](https://huggingface.co/papers/2510.12323)

RAG-Anything is a unified framework that enhances multimodal knowledge retrieval by integrating cross-modal relationships and semantic matching, outperforming existing methods on complex benchmarks.

### 7. Detect Anything via Next Point Prediction

[Read Paper](https://huggingface.co/papers/2510.12798)

Object detection has long been dominated by traditional coordinate
regression-based models, such as YOLO, DETR, and Grounding DINO. Although
recent efforts have attempted to leverage MLLMs to tackle this task, they face
challenges like low recall rate, duplicate predictions, coordinate
misalignment, etc. In this work, we bridge this gap and propose Rex-Omni, a
3B-scale MLLM that achieves state-of-the-art object perception performance. On
benchmarks like COCO and LVIS, Rex-Omni attains performance comparable to or
exceeding regression-based models (e.g., DINO, Grounding DINO) in a zero-shot
setting. This is enabled by three key designs: 1) Task Formulation: we use
special tokens to represent quantized coordinates from 0 to 999, reducing the
model's learning difficulty and improving token efficiency for coordinate
prediction; 2) Data Engines: we construct multiple data engines to generate
high-quality grounding, referring, and pointing data, providing semantically
rich supervision for training; \3) Training Pipelines: we employ a two-stage
training process, combining supervised fine-tuning on 22 million data with
GRPO-based reinforcement post-training. This RL post-training leverages
geometry-aware rewards to effectively bridge the discrete-to-continuous
coordinate prediction gap, improve box accuracy, and mitigate undesirable
behaviors like duplicate predictions that stem from the teacher-guided nature
of the initial SFT stage. Beyond conventional detection, Rex-Omni's inherent
language understanding enables versatile capabilities such as object referring,
pointing, visual prompting, GUI grounding, spatial referring, OCR and
key-pointing, all systematically evaluated on dedicated benchmarks. We believe
that Rex-Omni paves the way for more versatile and language-aware visual
perception systems.

### 8. A Survey of Vibe Coding with Large Language Models

[Read Paper](https://huggingface.co/papers/2510.12399)

The advancement of large language models (LLMs) has catalyzed a paradigm
shift from code generation assistance to autonomous coding agents, enabling a
novel development methodology termed "Vibe Coding" where developers validate
AI-generated implementations through outcome observation rather than
line-by-line code comprehension. Despite its transformative potential, the
effectiveness of this emergent paradigm remains under-explored, with empirical
evidence revealing unexpected productivity losses and fundamental challenges in
human-AI collaboration. To address this gap, this survey provides the first
comprehensive and systematic review of Vibe Coding with large language models,
establishing both theoretical foundations and practical frameworks for this
transformative development approach. Drawing from systematic analysis of over
1000 research papers, we survey the entire vibe coding ecosystem, examining
critical infrastructure components including LLMs for coding, LLM-based coding
agent, development environment of coding agent, and feedback mechanisms. We
first introduce Vibe Coding as a formal discipline by formalizing it through a
Constrained Markov Decision Process that captures the dynamic triadic
relationship among human developers, software projects, and coding agents.
Building upon this theoretical foundation, we then synthesize existing
practices into five distinct development models: Unconstrained Automation,
Iterative Conversational Collaboration, Planning-Driven, Test-Driven, and
Context-Enhanced Models, thus providing the first comprehensive taxonomy in
this domain. Critically, our analysis reveals that successful Vibe Coding
depends not merely on agent capabilities but on systematic context engineering,
well-established development environments, and human-agent collaborative
development models.

### 9. FlashVSR: Towards Real-Time Diffusion-Based Streaming Video
  Super-Resolution

[Read Paper](https://huggingface.co/papers/2510.12747)

Diffusion models have recently advanced video restoration, but applying them
to real-world video super-resolution (VSR) remains challenging due to high
latency, prohibitive computation, and poor generalization to ultra-high
resolutions. Our goal in this work is to make diffusion-based VSR practical by
achieving efficiency, scalability, and real-time performance. To this end, we
propose FlashVSR, the first diffusion-based one-step streaming framework
towards real-time VSR. FlashVSR runs at approximately 17 FPS for 768x1408
videos on a single A100 GPU by combining three complementary innovations: (i) a
train-friendly three-stage distillation pipeline that enables streaming
super-resolution, (ii) locality-constrained sparse attention that cuts
redundant computation while bridging the train-test resolution gap, and (iii) a
tiny conditional decoder that accelerates reconstruction without sacrificing
quality. To support large-scale training, we also construct VSR-120K, a new
dataset with 120k videos and 180k images. Extensive experiments show that
FlashVSR scales reliably to ultra-high resolutions and achieves
state-of-the-art performance with up to 12x speedup over prior one-step
diffusion VSR models. We will release the code, pretrained models, and dataset
to foster future research in efficient diffusion-based VSR.

### 10. Dr.LLM: Dynamic Layer Routing in LLMs

[Read Paper](https://huggingface.co/papers/2510.12773)

Large Language Models (LLMs) process every token through all layers of a
transformer stack, causing wasted computation on simple queries and
insufficient flexibility for harder ones that need deeper reasoning.
Adaptive-depth methods can improve efficiency, but prior approaches rely on
costly inference-time search, architectural changes, or large-scale retraining,
and in practice often degrade accuracy despite efficiency gains. We introduce
Dr.LLM, Dynamic routing of Layers for LLMs, a retrofittable framework that
equips pretrained models with lightweight per-layer routers deciding to skip,
execute, or repeat a block. Routers are trained with explicit supervision:
using Monte Carlo Tree Search (MCTS), we derive high-quality layer
configurations that preserve or improve accuracy under a compute budget. Our
design, windowed pooling for stable routing, focal loss with class balancing,
and bottleneck MLP routers, ensures robustness under class imbalance and long
sequences. On ARC (logic) and DART (math), Dr.LLM improves accuracy by up to
+3.4%p while saving 5 layers per example on average. Routers generalize to
out-of-domain tasks (MMLU, GSM8k, AIME, TruthfulQA, SQuADv2, GPQA, PIQA,
AGIEval) with only 0.85% accuracy drop while retaining efficiency, and
outperform prior routing methods by up to +7.7%p. Overall, Dr.LLM shows that
explicitly supervised routers retrofit frozen LLMs for budget-aware,
accuracy-driven inference without altering base weights.
