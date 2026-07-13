---
layout: post
title: "Daily AI Research Papers - Monday, July 13, 2026"
date: 2026-07-13
---

Keywords: text-to-image models, dense prediction, VAE latent space, DiT, task LoRA, token-local linear head, FLUX-Klein, trimap-free matting

---

### 1. Long-Horizon-Terminal-Bench: Testing the Limits of Agents on Long-Horizon Terminal Tasks with Dense Reward-Based Grading

[Read Paper](https://huggingface.co/papers/2607.08964)

AI agents have become capable of autonomously completing short, well-specified tasks. However, existing terminal benchmarks largely focus on simple problems that finish within minutes and are evaluated only by their final outcome. This setup overlooks intermediate progress and partial solutions, yielding sparse reward signals and an incomplete picture of agent capability. We introduce Long-Horizon-Terminal-Bench, a terminal benchmark of 46 long-horizon tasks spanning nine categories, including experiment reproduction, software engineering, multimodal analysis, interactive games, and scientific computing. Each task follows a Terminal-Bench-style setup with a reference solution or simulation engine, but is further decomposed into fine-grained graded subtasks. This design enables dense intermediate rewards and partial credit, allowing evaluation to capture not only whether an agent reaches the final goal, but also how far it progresses on open-ended workflows. Tasks in Long-Horizon-Terminal-Bench typically require hundreds of episodes and minutes to hours of execution, stressing long-horizon planning, long-context management, and iterative debugging rather than one-shot problem solving. We evaluate 15 frontier models and find that agents consume on average 9.9M tokens per task, with roughly 231 episodes and 85.3 minutes of execution time per run, making Long-Horizon-Terminal-Bench more demanding than prior terminal-based benchmarks. Even the strongest tested model achieves 15.2% pass@1 at a partial-reward threshold of 0.95 and 10.9% at a perfect-reward threshold of 1.0, while the mean pass rate across models is 4.3% and 1.7% under the two thresholds, respectively. These results reveal headroom for improvement. We further analyze failure modes and error patterns, and release Long-Horizon-Terminal-Bench to support future progress on long-horizon terminal agents.

### 2. Scalable Visual Pretraining for Language Intelligence

[Read Paper](https://huggingface.co/papers/2607.09657)

The rapid progress of large foundation models has been driven predominantly by pretraining on large-scale text corpora. However, many forms of knowledge are conveyed through visual representations, where figures, typeset equations, and page layouts carry rich information that cannot be faithfully or completely captured by text alone. Yet current pretraining approaches discard these visual cues by converting visually rich sources, such as documents and web pages, into plain text for learning language intelligence. This paper challenges the default assumption that language models must be trained on text-only representations and shows that Visual Pretraining is a scalable learner for foundation model intelligence. To this end, we conduct a systematic study of unsupervised visual pretraining paradigms that directly leverage visual documents without text extraction. Across multiple backbones and benchmarks, visual pretraining on the same underlying corpora consistently outperforms text-only pretraining, offering an efficient pathway to scalable language intelligence.

### 3. Video Generation Models are General-Purpose Vision Learners

[Read Paper](https://huggingface.co/papers/2607.09024)

Driven by next-token prediction, NLP shifted from task-specific models into powerful generalist foundation models. What, then, is the equivalent catalyst needed to achieve a general-purpose model in computer vision? In this paper, we contend that large-scale text-to-video generation serves as a strong pre-training paradigm for computer vision, providing the necessary spatiotemporal priors, vision-language alignment, and scalability required for general visual intelligence. We introduce GenCeption, which leverages a pre-trained video generative diffusion backbone to define a feed-forward perception model, capable of performing various vision tasks steered by text instructions. Empirical results demonstrate that GenCeption achieves state-of-the-art performance across a diverse suite of tasks, including depth, surface normal, and camera pose estimation, expression-referring segmentation, and 3D keypoint prediction, often matching or surpassing specialized models (e.g. DepthAnything3, SAM3, D4RT, VGGT-Omega, Sapiens, David, Genmo, and Lotus-2). Furthermore, the video generative pretrained backbone outperforms alternative pretraining paradigms (e.g., V-JEPA, and Video MAE) under comparable settings. Importantly, GenCeption exhibits preliminary data and model scaling properties along with exceptional data efficiency, where it achieves comparable performance with leading models like D4RT and VGGT-Omega with 7 to 500 less training data. Finally, GenCeption also exhibits intriguing emergent behaviors: a model trained exclusively on synthetic human videos generalizes to real-world footage and out-of-distribution object categories (e.g., animals and robots). These findings suggest that video generation is not merely a synthesis tool, but a foundational path toward generalist vision intelligence for the physical world. Project page: https://genception.github.io

### 4. Trust Region Policy Distillation

[Read Paper](https://huggingface.co/papers/2607.04751)

Big goals are hard to achieve all at once; breaking them into small steps is wiser. We present Trust Region Policy Distillation (TOP-D), which transforms the notoriously unstable, high-variance On-Policy Distillation (OPD) into a stable training paradigm by dynamically constructing a proximal teacher. Theoretically, we establish a rigorous framework demonstrating that TOP-D inherently controls gradient variance. By providing a formal global convergence analysis alongside a monotonic improvement bound, we mathematically formalize the reliability and stability of the overall training dynamics. Empirically, TOP-D dramatically enhances training stability, sample efficiency, and final performance on mathematical reasoning tasks. More importantly, TOP-D introduces zero additional computational overhead, positioning itself as a promising alternative to the well-established OPD paradigm.

### 5. KronQ: LLM Quantization via Kronecker-Factored Hessian

[Read Paper](https://huggingface.co/papers/2607.07964)

Post-training quantization (PTQ) is a widely adopted technique for compressing large language models (LLMs) without retraining. Existing second-order PTQ methods, including GPTQ, construct quantization objectives exclusively from input activation statistics, effectively assuming that all output channels contribute equally to the layer-wise reconstruction objective. We propose KronQ, a PTQ framework that challenges this assumption by introducing the gradient covariance into the quantization pipeline. Under the Kronecker-factored Hessian approximation, the quantization loss depends jointly on both the activation and gradient covariances, and KronQ exploits this at two complementary levels. (1) KronQ introduces bidirectional incoherence processing, extending the existing input-side random rotation to the output dimension using the gradient covariance, reducing weight magnitude variance across both input and output dimensions. (2) KronQ derives a new sensitivity metric for inter-layer mixed-precision allocation, driven by the gradient and activation Hessian traces. Notably, in the case of 2-bit weight-only quantization on LLaMA-3-70B, while GPTQ and GPTAQ diverge or produce degenerate quantizations (>2000 perplexity on WikiText-2), KronQ achieves 7.93 perplexity.

### 6. From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models

[Read Paper](https://huggingface.co/papers/2607.06553)

Pretrained diffusion transformers can be adapted for dense prediction tasks by mapping tokens to task-native outputs instead of generating RGB images, achieving state-of-the-art results with minimal additional parameters.

### 7. PanoWorld: Real-World Panoramic Generation

[Read Paper](https://huggingface.co/papers/2607.09661)

In this work, we aim to address the challenge of long-range memory in panoramic world models by exploiting the rotation-equivariant property of omnidirectional representations, where rotation can be treated as an implicit geometric transformation.Building on this insight, we propose PanoWorld, which simplifies camera trajectories into translations via fixed headings for both current-action modeling and long-range memory through Dense Panoramic Ray-Conditioning (DPRC) and Geometry-aware Memory Augmentation (GMA).Then, a three-stage training pipeline is introduced to progressively optimize each component. To better evaluate physical consistency under large-scale spatial variations and diverse illumination conditions, where existing datasets are relatively stable, we construct World360, a large-scale dataset consisting of both real-world video clips collected via panoramic unmanned aerial vehicles and high-quality simulated clips generated by AirSim360.Extensive experiments on World360 demonstrate the effectiveness of PanoWorld, outperforming alternative methods by a large margin.Our models, training code, and dataset will be publicly available. More information can be found on our project page: https://lihaoy-ux.github.io/panoworld-page/.

### 8. Self-Guided Test-Time Training for Long-Context LLMs

[Read Paper](https://huggingface.co/papers/2607.09415)

Long-context processing has become increasingly important for large language models (LLMs), but simply extending the context window does not guarantee effective utilization of long inputs. As input length grows, accuracy often degrades, indicating that models still struggle to identify and use the evidence most relevant to a question. A promising way to improve long-context utilization is test-time training (TTT), which treats the test context as a training example for instance-specific parameter adaptation. However, applying TTT to the entire long context is prohibitively expensive, while adapting on randomly sampled spans introduces severe noise. Because most spans in a long context are irrelevant to the specific question, training on them may even degrade the base model's performance. Our preliminary study shows that TTT is highly sensitive to training-span quality: on LongBench-v2, TTT on randomly sampled spans hurts performance, whereas TTT on oracle spans substantially improves it. Motivated by this, we propose a simple method, Self-Guided TTT (S-TTT): before adaptation, the model identifies the evidence spans it should learn from, and the standard language-modeling training objective is applied only to those selected spans. On two challenging long-context reasoning benchmarks, LongBench-v2 and LongBench-Pro, S-TTT improves accuracy for both Qwen3-4B-Thinking-2507 and Llama-3.1-8B-Instruct, achieving up to a 15% relative improvement.

### 9. Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning

[Read Paper](https://huggingface.co/papers/2607.08393)

Fine-tuning LLMs to inject new knowledge faces a critical challenge: LLMs can quickly memorize new facts, yet fail to use them for downstream reasoning tasks. We formalize this failure as the \textbf{Knowing--Using Gap}, characterized by an accuracy gap and a temporal lag between memorization and generalization. To understand this phenomenon, we fine-tune LLMs with unseen knowledge and monitor the spatial permeation dynamics of the knowledge internally using a novel intervention technique called self-patching. Self-patching identifies activation locations where relocating representations substantially improves failed generalization cases. These results are consistent with a knowledge-circuit misalignment hypothesis: memorized representations can exist internally but may not be routed to computation-effective layers. To demonstrate the practicality of this diagnostic finding, we design a simple heuristic strategy which recovers 58--75\% of the oracle headroom in generalization failure. Experiments are done cross-domain for the robustness of this finding.

### 10. Flow-ERD: Agent-type Aware Flow Matching with Entropy-Regularized Distillation for Diverse Traffic Simulation

[Read Paper](https://huggingface.co/papers/2607.06957)

Flow-ERD is a multi-agent traffic simulator that combines agent-type aware flow matching with entropy-regularized distillation to achieve both realistic and diverse motion patterns.
