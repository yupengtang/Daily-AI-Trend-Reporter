---
layout: post
title: "Daily AI Research Papers - Monday, February 16, 2026"
date: 2026-02-16
---

Keywords: 3D RoPE, Codec Patchification, Feature Activation Coverage, GUI agents, LLM, Multimodal Large Language Models, SimMotion benchmarks, Thinking-with-Images

---

### 1. SQuTR: A Robustness Benchmark for Spoken Query to Text Retrieval under Acoustic Noise

[Read Paper](https://huggingface.co/papers/2602.12783)

Spoken query retrieval is an important interaction mode in modern information retrieval. However, existing evaluation datasets are often limited to simple queries under constrained noise conditions, making them inadequate for assessing the robustness of spoken query retrieval systems under complex acoustic perturbations. To address this limitation, we present SQuTR, a robustness benchmark for spoken query retrieval that includes a large-scale dataset and a unified evaluation protocol. SQuTR aggregates 37,317 unique queries from six commonly used English and Chinese text retrieval datasets, spanning multiple domains and diverse query types. We synthesize speech using voice profiles from 200 real speakers and mix 17 categories of real-world environmental noise under controlled SNR levels, enabling reproducible robustness evaluation from quiet to highly noisy conditions. Under the unified protocol, we conduct large-scale evaluations on representative cascaded and end-to-end retrieval systems. Experimental results show that retrieval performance decreases as noise increases, with substantially different drops across systems. Even large-scale retrieval models struggle under extreme noise, indicating that robustness remains a critical bottleneck. Overall, SQuTR provides a reproducible testbed for benchmarking and diagnostic analysis, and facilitates future research on robustness in spoken query to text retrieval.

### 2. Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs

[Read Paper](https://huggingface.co/papers/2602.10388)

Feature Activation Coverage measures data diversity in an interpretable feature space and enables diversity-driven data synthesis that improves downstream performance across multiple language model architectures.

### 3. MedXIAOHE: A Comprehensive Recipe for Building Medical MLLMs

[Read Paper](https://huggingface.co/papers/2602.12705)

MedXIAOHE is a medical vision-language foundation model that enhances clinical understanding through entity-aware continual pretraining, reinforcement learning, and tool-augmented agentic training for reliable diagnostic reasoning.

### 4. Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception

[Read Paper](https://huggingface.co/papers/2602.11858)

Region-to-Image Distillation enables fine-grained visual perception in MLLMs by training models to internally perform iterative zooming during inference, eliminating the need for repeated tool calls and visual re-encoding while maintaining high performance across multiple benchmarks.

### 5. Towards Universal Video MLLMs with Attribute-Structured and Quality-Verified Instructions

[Read Paper](https://huggingface.co/papers/2602.13013)

A large-scale dataset and model for fine-grained audiovisual understanding are introduced, demonstrating improved caption quality and reduced hallucinations through structured annotations and supervised fine-tuning.

### 6. OneVision-Encoder: Codec-Aligned Sparsity as a Foundational Principle for Multimodal Intelligence

[Read Paper](https://huggingface.co/papers/2602.08683)

Visual understanding can be improved by aligning architectures with information-theoretic principles of video compression, using sparsity-driven encoding that outperforms traditional approaches in efficiency and accuracy.

### 7. CoPE-VideoLM: Codec Primitives For Efficient Video Language Models

[Read Paper](https://huggingface.co/papers/2602.13191)

Video Language Models (VideoLMs) empower AI systems to understand temporal dynamics in videos. To fit to the maximum context window constraint, current methods use keyframe sampling which can miss both macro-level events and micro-level details due to the sparse temporal coverage. Furthermore, processing full images and their tokens for each frame incurs substantial computational overhead. To address these limitations, we propose to leverage video codec primitives (specifically motion vectors and residuals) which natively encode video redundancy and sparsity without requiring expensive full-image encoding for most frames. To this end, we introduce lightweight transformer-based encoders that aggregate codec primitives and align their representations with image encoder embeddings through a pre-training strategy that accelerates convergence during end-to-end fine-tuning. Our approach reduces the time-to-first-token by up to 86% and token usage by up to 93% compared to standard VideoLMs. Moreover, by varying the keyframe and codec primitive densities we are able to maintain or exceed performance on 14 diverse video understanding benchmarks spanning general question answering, temporal reasoning, long-form understanding, and spatial scene understanding.

### 8. SemanticMoments: Training-Free Motion Similarity via Third Moment Features

[Read Paper](https://huggingface.co/papers/2602.09146)

Temporal statistics in semantic feature space provide a scalable approach for motion-centric video understanding, outperforming existing RGB, flow, and text-supervised methods.

### 9. GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics

[Read Paper](https://huggingface.co/papers/2602.12617)

GeoAgent achieves superior geolocation reasoning performance through a specialized dataset and reward mechanisms that ensure geographic accuracy and reasoning consistency.

### 10. What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis

[Read Paper](https://huggingface.co/papers/2602.12395)

Reinforcement learning (RL) with verifiable rewards has become a standard post-training stage for boosting visual reasoning in vision-language models, yet it remains unclear what capabilities RL actually improves compared with supervised fine-tuning as cold-start initialization (IN). End-to-end benchmark gains conflate multiple factors, making it difficult to attribute improvements to specific skills. To bridge the gap, we propose a Frankenstein-style analysis framework including: (i) functional localization via causal probing; (ii) update characterization via parameter comparison; and (iii) transferability test via model merging. Instead, RL induces a consistent inference-time shift primarily in mid-to-late layers, and these mid-to-late refinements are both transferable (via merging) and necessary (via freezing) for RL gains. Overall, our results suggest that RL's reliable contribution in visual reasoning is not a uniform enhancement of visual perception, but a systematic refinement of mid-to-late transformer computation that improves vision-to-reasoning alignment and reasoning performance, highlighting the limitations of benchmark-only evaluation for understanding multimodal reasoning improvements.
