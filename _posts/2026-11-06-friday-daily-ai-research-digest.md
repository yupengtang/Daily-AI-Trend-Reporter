---
layout: post
title: "Daily AI Research Papers - Friday, November 06, 2026"
date: 2026-11-06
---

Keywords: reinforcement learning, policy training, rollouts, trajectories, LLMs, Adaptive Harness Router, Top-k variants, Harness Behavior Profiles

---

### 1. VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding

[Read Paper](https://huggingface.co/papers/2611.08449)

Models for physical AI tasks, such as robotics and autonomous driving, need not reason solely in language. They can use visual plans--points, bounding boxes, and trajectories--to represent spatial goals and anticipate motion. However, autoregressive vision-language models generate these plans one coordinate token at a time, making richer predictions increasingly costly. We introduce VFlash, a method for fast, accurate visual planning through Flash Primitive Decoding. VFlash exploits the known structure of visual primitives to predict their coordinates jointly, using a lightweight drafter that generates an entire primitive in one forward pass. The layout adapts to the primitive's type, dimensionality, and length, supporting compact boxes and long, variable-length trajectories. Crucially, spatial accuracy need not require reproducing an autoregressive model's exact token sequence. VFlash therefore trains on annotated primitives and directly accepts structurally valid predictions without target-token agreement. Across the evaluated 2D and 3D robot trajectory tasks, VFlash consistently reduces prediction errors relative to autoregressive decoding. Preliminary single-rollout measurements indicate approximately 25x faster generation for 3D point motion forecasting. A world action model conditioned on VFlash plans achieves 70.2% average success across four MolmoSpaces task suites, compared with 62.6% for the baseline.

### 2. Harness-S: Learning from Sibling Harnesses for Generalizable Agent Training

[Read Paper](https://huggingface.co/papers/2611.04277)

A selective harness-rescue training method uses specialized sibling harnesses to replace failed reinforcement learning rollouts and transfer their benefits into a single policy that performs better under the original base harness.

### 3. Random Is Almost Enough? Stable and Low-Cost Benchmark Compression via BestRand

[Read Paper](https://huggingface.co/papers/2611.00883)

Benchmark compression reduces the often prohibitive time and cost of model evaluation by evaluating models on a small benchmark subset that closely matches full-benchmark results. To quantify estimation error, we derive an accurate analytical approximation to the error distribution when estimating full-benchmark scores from a subset, showing that most subsets yield small errors, while a few produce much larger errors, resulting in a long-tailed distribution. This means random sampling alone can achieve surprisingly strong performance with high probability, but remains unstable due to rare large errors. We further show that this instability is inherent to modern foundation models and persists even as more models are evaluated. Based on these findings, we introduce BestRand, a stable and low-cost benchmark compression method that preserves the efficiency of random sampling while substantially reducing its high-error tail. We validate BestRand on 70+ text, multimodal, and agent benchmarks, and further extend it to VLA evaluation. Compared with competitive methods, BestRand reduces high-error risk by 50.2% and estimation error by 16.4% on average, while requiring only 51.3% of the construction cost. It further reduces OOD estimation error by 21.8% on average across score-range, model-family, and temporal shifts. Together, our theory provides practical guidance for understanding and predicting the accuracy and reliability of benchmark compression, while BestRand enables stable and accurate compression at low cost. The code and data are publicly available onhttps://github.com/George-Ling3/BestRand.

### 4. SLA: Scaling Spatial-Language-Action Model via Continual Pre-training

[Read Paper](https://huggingface.co/papers/2611.08373)

SLA is a spatial-language-action model that continually pre-trains and post-trains a pretrained VLA policy with spatial alignment to a frozen 3D perception model, using feature-dependent gates and probabilistic state dropout to improve spatial understanding across diverse robot data while preserving inference architecture.

### 5. WeLM-Image: A Unified Image Generation Model towards Visual Intelligence

[Read Paper](https://huggingface.co/papers/2611.07392)

We present WeLM-Image, a unified image foundation model that integrates multimodal understanding into text-to-image generation, instruction-guided editing, and multi-reference synthesis. Designed for single-GPU deployment in WeChat's large-scale consumer applications, the model processes text, reference images, and target latents as one interleaved sequence. A Mixture-of-Transformers (MoT) backbone combines shared attention with independent understanding and generation parameters to limit optimization interference. Equipping each path with Mixture-of-Experts (MoE) modules provides sparse capacity, with 31.7B stored parameters and approximately 5.4B generation parameters activated per target latent token. A generation-oriented data engine combines a knowledge-grounded visual vocabulary with large-scale visual clustering to balance semantic coverage and editing tasks, while training-time evaluation guides iterative data refinement. Bilingual annotations and verified editing supervision support progressive training up to native 1.5K resolution. We optimize generation and editing separately with rubric- and checklist-based rewards, then consolidate the specialized policies through noise-aware on-policy distillation. In the reported overall comparisons, the 1536-resolution model ranks first among the evaluated open-source models on PRISM-Bench, Qwen-Image-Bench, and UnicBench. MeanFlow-based few-step distillation yields WeLM-Image-Turbo, which generates a 1024times1024 image in 4.98\,s on a single NVIDIA H20 GPU using eight guidance-free steps.

### 6. TailSteer: Trajectory-Tail Steering for Drift-Resistant Multi-turn Image Editing

[Read Paper](https://huggingface.co/papers/2611.06741)

TailSteer, a final-stage training framework, reduces cumulative visual drift in multi-turn editing by keeping early denoising dynamics close to the base model and learning tail-focused preservation corrections from on-policy rollouts, preserving pretrained generative and editing abilities.

### 7. EvoMemState: Evolving Memory State for Personal Agents

[Read Paper](https://huggingface.co/papers/2611.03743)

Long-term memory enables large language model (LLM) agents to provide continuous personalized assistance by retaining user information across interactions. Existing systems support this process through fact extraction, organization, updating, and retrieval, but largely maintain memories as independent facts. This fact-centric paradigm may overlook how new facts update, constrain, or conflict with earlier ones, while leaving their applicability, role, and priority for the current request implicit. In this paper, we introduce EvoMemState, a state-centric framework that augments persistent facts with two complementary states: Relational States that capture how facts remain valid and relate to one another as memory evolves, and a Reasoning State that determines how these facts should be applied to the current request. Specifically, EvoMemState consists of three key components: (i) Ontology-Guided Memory State Induction extracts Persistent Facts and constructs Relational States that maintain their validity, scope, and evolving relations. (ii) RL-Based State Evolution internalizes memory-use experience into a policy through reinforcement learning to generate a request-specific Reasoning State. (iii) State-Aware Reasoning then combines the relevant facts with their Relational States and the generated Reasoning State to guide the personal agent. Extensive experiments on long-term dialogue and memory-use benchmarks demonstrate consistent improvements over existing memory systems in retrieval completeness and memory-grounded response accuracy. Our code and implementation details are available at https://github.com/XMUDeepLIT/EvoMemState.git.

### 8. VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents

[Read Paper](https://huggingface.co/papers/2611.04894)

VisSearch-Bench evaluates multimodal agents on visually intensive search tasks that require integrating visual context with external information, showing that even the best system fully solves only 26.47% of tasks.

### 9. Towards Foundation Decision Models: Drex

[Read Paper](https://huggingface.co/papers/2611.07854)

Drex open decision models return probability distributions over user-supplied options without autoregressive decoding, using supervised fine-tuning, RLAF, and consistency losses to improve accuracy, robustness, and latency.

### 10. Open-HumanScale: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models

[Read Paper](https://huggingface.co/papers/2611.06817)

Egocentric human video offers a scalable source of manipulation priors for robot learning, yet translating raw data scale into generalizable robot policies demands rigorous data curation and training recipes. We introduce Open-HumanScale, the first fully open recipe for scaling egocentric pretraining to 50,000 hours, anchored by Open-HumanScale-50K---a curated egocentric video corpus densely annotated with hand motion and natural language. Leveraging this corpus, we systematically investigate pure human pretraining via closed-loop simulation evaluations and uncover an empirical log-linear scaling law linking pretraining data hours to downstream robot success rates. Grounded in these findings, we develop HumanWAM-Vanilla, a world--action model pretrained exclusively on 50k hours of egocentric human video. Upon robot post-training, HumanWAM-Vanilla reaches 92.08% success on RoboTwin 2.0 Clean2Clean and 68.88% on Clean2Random, alongside a 14.17% mean success rate and a 19.00 mean progress score on RoboDojo. To push the capability frontier further, we explore joint human--robot pretraining with HumanWAM-Goal, a visual-goal-conditioned world--action model. Hybrid data scaling unlocks broad zero-shot generalization across both simulation and physical benchmarks, propelling post-trained RoboTwin 2.0 performance to new state-of-the-art levels: 92.70% on Clean2Clean and a substantial leap to 84.60% on Clean2Random. On physical robots, HumanWAM-Goal further exhibits emergent embodied in-context learning, dynamically adapting its manipulation behavior to unseen visual goals without parameter updates. To support reproducible scaling research in embodied AI, we open-source the Open-HumanScale-50K corpus, data curation and pretraining pipelines, alongside the pretrained and post-trained HumanWAM checkpoints, inference code, and simulation evaluation suites.
