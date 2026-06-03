---
layout: post
title: "Daily AI Research Papers - Monday, December 29, 2025"
date: 2025-12-29
---

Keywords: Retrieval-Augmented Generation, LLM-based RAG systems, global context awareness, hierarchical summarization, mindscape, query embeddings, coherent global representation, long-context retrieval

---

### 1. Mindscape-Aware Retrieval Augmented Generation for Improved Long Context Understanding

[Read Paper](https://huggingface.co/papers/2512.17220)

MiA-RAG enhances long-context reasoning in LLM-based RAG systems by incorporating hierarchical summarization to create global semantic representations that guide both retrieval and generation processes.

### 2. InsertAnywhere: Bridging 4D Scene Geometry and Diffusion Models for Realistic Video Object Insertion

[Read Paper](https://huggingface.co/papers/2512.17504)

InsertAnywhere framework enhances video object insertion by generating geometrically consistent and visually coherent scenarios through 4D aware mask generation and diffusion-based synthesis.

### 3. MAI-UI Technical Report: Real-World Centric Foundation GUI Agents

[Read Paper](https://huggingface.co/papers/2512.22047)

The development of GUI agents could revolutionize the next generation of human-computer interaction. Motivated by this vision, we present MAI-UI, a family of foundation GUI agents spanning the full spectrum of sizes, including 2B, 8B, 32B, and 235B-A22B variants. We identify four key challenges to realistic deployment: the lack of native agent-user interaction, the limits of UI-only operation, the absence of a practical deployment architecture, and brittleness in dynamic environments. MAI-UI addresses these issues with a unified methodology: a self-evolving data pipeline that expands the navigation data to include user interaction and MCP tool calls, a native device-cloud collaboration system routes execution by task state, and an online RL framework with advanced optimizations to scale parallel environments and context length. MAI-UI establishes new state-of-the-art across GUI grounding and mobile navigation. On grounding benchmarks, it reaches 73.5% on ScreenSpot-Pro, 91.3% on MMBench GUI L2, 70.9% on OSWorld-G, and 49.2% on UI-Vision, surpassing Gemini-3-Pro and Seed1.8 on ScreenSpot-Pro. On mobile GUI navigation, it sets a new SOTA of 76.7% on AndroidWorld, surpassing UI-Tars-2, Gemini-2.5-Pro and Seed1.8. On MobileWorld, MAI-UI obtains 41.7% success rate, significantly outperforming end-to-end GUI models and competitive with Gemini-3-Pro based agentic frameworks. Our online RL experiments show significant gains from scaling parallel environments from 32 to 512 (+5.2 points) and increasing environment step budget from 15 to 50 (+4.3 points). Finally, the native device-cloud collaboration system improves on-device performance by 33%, reduces cloud model calls by over 40%, and preserves user privacy.

### 4. Masking Teacher and Reinforcing Student for Distilling Vision-Language Models

[Read Paper](https://huggingface.co/papers/2512.22238)

Large-scale vision-language models (VLMs) have recently achieved remarkable multimodal understanding, but their massive size makes them impractical for deployment on mobile or edge devices. This raises the need for compact yet capable VLMs that can efficiently learn from powerful large teachers. However, distilling knowledge from a large teacher to a small student remains challenging due to their large size gap: the student often fails to reproduce the teacher's complex, high-dimensional representations, leading to unstable learning and degraded performance. To address this, we propose Masters (Masking Teacher and Reinforcing Student), a mask-progressive reinforcement learning (RL) distillation framework. Masters first masks non-dominant weights of the teacher to reduce unnecessary complexity, then progressively restores the teacher by gradually increasing its capacity during training. This strategy allows the student to learn richer representations from the teacher in a smooth and stable manner. To further refine knowledge transfer, Masters integrates an offline RL stage with two complementary rewards: an accuracy reward that measures the correctness of the generated responses, and a distillation reward that quantifies the ease of transferring responses from teacher to student. Unlike online think-answer RL paradigms that are computationally expensive and generate lengthy responses, our offline RL leverages pre-generated responses from masked teachers. These provide rich yet efficient guidance, enabling students to achieve strong performance without requiring the think-answer process.

### 5. UniPercept: Towards Unified Perceptual-Level Image Understanding across Aesthetics, Quality, Structure, and Texture

[Read Paper](https://huggingface.co/papers/2512.21675)

Multimodal large language models (MLLMs) have achieved remarkable progress in visual understanding tasks such as visual grounding, segmentation, and captioning. However, their ability to perceive perceptual-level image features remains limited. In this work, we present UniPercept-Bench, a unified framework for perceptual-level image understanding across three key domains: Aesthetics, Quality, Structure and Texture. We establish a hierarchical definition system and construct large-scale datasets to evaluate perceptual-level image understanding. Based on this foundation, we develop a strong baseline UniPercept trained via Domain-Adaptive Pre-Training and Task-Aligned RL, enabling robust generalization across both Visual Rating (VR) and Visual Question Answering (VQA) tasks. UniPercept outperforms existing MLLMs on perceptual-level image understanding and can serve as a plug-and-play reward model for text-to-image generation. This work defines Perceptual-Level Image Understanding in the era of MLLMs and, through the introduction of a comprehensive benchmark together with a strong baseline, provides a solid foundation for advancing perceptual-level multimodal image understanding.

### 6. TimeBill: Time-Budgeted Inference for Large Language Models

[Read Paper](https://huggingface.co/papers/2512.21859)

Large Language Models (LLMs) are increasingly deployed in time-critical systems, such as robotics, autonomous driving, embodied intelligence, and industrial automation, where generating accurate responses within a given time budget is crucial for decision-making, control, or safety-critical tasks. However, the auto-regressive generation process of LLMs makes it challenging to model and estimate the end-to-end execution time. Furthermore, existing efficient inference methods based on a fixed key-value (KV) cache eviction ratio struggle to adapt to varying tasks with diverse time budgets, where an improper eviction ratio may lead to incomplete inference or a drop in response performance. In this paper, we propose TimeBill, a novel time-budgeted inference framework for LLMs that balances the inference efficiency and response performance. To be more specific, we propose a fine-grained response length predictor (RLP) and an execution time estimator (ETE) to accurately predict the end-to-end execution time of LLMs. Following this, we develop a time-budgeted efficient inference approach that adaptively adjusts the KV cache eviction ratio based on execution time prediction and the given time budget. Finally, through extensive experiments, we demonstrate the advantages of TimeBill in improving task completion rate and maintaining response performance under various overrun strategies.

### 7. ProEdit: Inversion-based Editing From Prompts Done Right

[Read Paper](https://huggingface.co/papers/2512.22118)

Inversion-based visual editing provides an effective and training-free way to edit an image or a video based on user instructions. Existing methods typically inject source image information during the sampling process to maintain editing consistency. However, this sampling strategy overly relies on source information, which negatively affects the edits in the target image (e.g., failing to change the subject's atributes like pose, number, or color as instructed). In this work, we propose ProEdit to address this issue both in the attention and the latent aspects. In the attention aspect, we introduce KV-mix, which mixes KV features of the source and the target in the edited region, mitigating the influence of the source image on the editing region while maintaining background consistency. In the latent aspect, we propose Latents-Shift, which perturbs the edited region of the source latent, eliminating the influence of the inverted latent on the sampling. Extensive experiments on several image and video editing benchmarks demonstrate that our method achieves SOTA performance. In addition, our design is plug-and-play, which can be seamlessly integrated into existing inversion and editing methods, such as RF-Solver, FireFlow and UniEdit.

### 8. See Less, See Right: Bi-directional Perceptual Shaping For Multimodal Reasoning

[Read Paper](https://huggingface.co/papers/2512.22120)

Large vision-language models (VLMs) often benefit from intermediate visual cues, either injected via external tools or generated as latent visual tokens during reasoning, but these mechanisms still overlook fine-grained visual evidence (e.g., polylines in charts), generalize poorly across domains, and incur high inference-time cost. In this paper, we propose Bi-directional Perceptual Shaping (BiPS), which transforms question-conditioned masked views into bidirectional where-to-look signals that shape perception during training. BiPS first applies a KL-consistency constraint between the original image and an evidence-preserving view that keeps only question-relevant regions, encouraging coarse but complete coverage of supporting pixels. It then applies a KL-separation constraint between the original and an evidence-ablated view where critical pixels are masked so the image no longer supports the original answer, discouraging text-only shortcuts (i.e., answering from text alone) and enforcing fine-grained visual reliance. Across eight benchmarks, BiPS boosts Qwen2.5-VL-7B by 8.2% on average and shows strong out-of-domain generalization to unseen datasets and image types.

### 9. Omni-Weather: Unified Multimodal Foundation Model for Weather Generation and Understanding

[Read Paper](https://huggingface.co/papers/2512.21643)

Omni-Weather is a multimodal foundation model that integrates weather generation and understanding using a shared self-attention mechanism and a Chain-of-Thought dataset to enable interpretable, high-quality outputs.

### 10. InSight-o3: Empowering Multimodal Foundation Models with Generalized Visual Search

[Read Paper](https://huggingface.co/papers/2512.18745)

O3-Bench evaluates multimodal reasoning with interleaved attention to visual details, while InSight-o3 uses a multi-agent framework to improve performance through specialized visual search and reasoning tasks.
