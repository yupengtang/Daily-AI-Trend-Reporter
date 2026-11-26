---
layout: post
title: "Daily AI Research Papers - Thursday, November 26, 2026"
date: 2026-11-26
---

Keywords: Looped Transformers, Mixture-of-Loops Transformer (MoLT), Mixture-of-Loops Attention (MoLA), keys and values (KV) memory, loop residual, MoLT-R, loop KV, MoLT-KV

---

### 1. Real2SimLab: A Unified Platform for Real-to-Simulation Transfer

[Read Paper](https://huggingface.co/papers/2611.34849)

Real-to-simulation transfer, which builds simulations from real-world observations, has long been studied in research and industry. Recently, coding agents have become capable of generating simulation programs from video recordings, and several benchmarks have been proposed to evaluate them. However, each individual benchmark defines its own inputs, deliverables, and scoring pipeline. This leads to N times M integrations when evaluating N methods on M benchmarks. To this end, we present Real2SimLab, a unified platform for real-to-simulation transfer. It evaluates a method by running its written program and scoring the output simulation with the original scorer of each benchmark. Moreover, Real2SimLab provides a pipeline to create training data and reward annotations for real-to-simulation methods. Our pipeline carefully avoids data contamination by filtering out the video recordings that exist in benchmarks. For each video recording, a program is first produced, and then our pipeline computes its reward with the same code that Real2SimLab uses for evaluation. So far, it supports 47 coding agents and 5 non-agent methods on 6 benchmarks. Both the platform and Awesome-Real2Sim, our research map of nearly 300 real-to-simulation papers, are open source, and we invite the Real2SimHub community to build them together.

### 2. Sage-Med: Scaling Medical Multimodal Intelligence

[Read Paper](https://huggingface.co/papers/2611.34575)

Establishing effective scaling laws for medical multimodal large language models (MLLMs) remains a persistent and fundamental challenge. Our systematic evaluation shows that existing medical MLLMs tend to underperform frontier general-purpose MLLMs across a range of medical tasks, while scaling model size primarily improves text capabilities rather than multimodal perception, and enabling reasoning modes fails to close this visual gap. To investigate how medical multimodal intelligence scales in practice, we introduce Sage-Med-Preview, a 35B-parameter medical multimodal model with 3B activated parameters, built by combining stage-specific curation over 322.7 million candidate records with a progressive training curriculum spanning medical pretraining, staged supervised fine-tuning, and adaptive reinforcement learning. Across this curriculum, progressive supervised refinement and adaptive reinforcement learning yield consistent stage-wise gains, substantially advancing both specialized visual understanding and medical language capabilities. Evaluated against 77 competing models across 238 clinical tasks spanning 61 public and 13 private multi-center benchmarks in more than 40 modalities, Sage-Med-Preview achieves an average score of 58.26% on the 31 public multimodal benchmarks. It outperforms the state-of-the-art open-source medical model Hulu-Med-235B-A22B at 50.99% and the general scientific model Intern-S2-Preview at 52.24%, and approaches the advanced proprietary model GPT-6.2-Sol across multiple tasks such as comprehensive multimodal understanding, report generation, and visual grounding. Although substantial challenges remain in complex medical reasoning and multi-step clinical workflows, this report provides empirical evidence, actionable data and training recipes, and a rigorous benchmark toward scalable clinical multimodal intelligence.

### 3. Mixture-of-Loops Transformer

[Read Paper](https://huggingface.co/papers/2611.34923)

A Mixture-of-Loops Transformer reuses key-value memory across loops and depths, and its U-shaped variant improves pretraining efficiency and inference latency.

### 4. DyLoMo-DiT: Dynamic Looped Mixture-of-Experts Diffusion Transformers

[Read Paper](https://huggingface.co/papers/2611.30844)

A dynamic looped mixture-of-experts diffusion transformer reuses shared parameters and varies compute across noise levels, improving image generation quality under matched budgets.

### 5. VibeVideoAgent: Closing the Perceptual Loop for Code-Driven Video Generation

[Read Paper](https://huggingface.co/papers/2611.34237)

Code-driven video generation asks a coding agent to write and run rendering programs. This gives exact control over typography, data visualizations, synchronized audio, and multi-shot scenes, and it leaves an editable project behind. A program that exits cleanly can still render overlapping labels, a camera aimed away from its subject, broken object interactions, or narration that drifts out of sync, and none of these raise a runtime error. Feeding rendered frames and audio back into the coding session bloats its long-horizon context, while critiquing only the finished full-resolution render turns early design mistakes into costly rewrites. We introduce VibeVideoAgent, a training-free framework that lets coding agents inspect their drafts throughout production. Multimodal perception runs in short-lived contexts kept apart from the coder and returns compact, grounded Findings. Seven primitives operate on a unified multi-clock representation of assets and drafts: they explore media from coarse to fine, verify quoted requirements with evidence from every required modality, compare a draft with a reference or an earlier revision on separate clocks, and measure stream signals locally. A lightweight staged protocol has the agent test its riskiest decision on a minimal preview first. On all 239 VibeVideoBench tasks, with matched agent runtimes and a fixed perception model, VibeVideoAgent improves holistic audiovisual quality (L3) for Claude Opus 5.5 (4.03 -> 4.43, while L2 rubric compliance rises from 98.87% to 99.35%), GPT-6 Astra (4.03 -> 4.27), and Claude Sonnet 5.5 (3.97 -> 4.10), with the largest gains in audio-visual synchronization (up to +0.56). Controlled ablations show that each component contributes, and trajectory audits show that the granularity of code edits governs the trade-off between rubric adherence and perceptual quality.

### 6. Improving and Expanding Video Editing Capabilities Without Videos

[Read Paper](https://huggingface.co/papers/2611.34941)

Image-to-Video Transfer improves pretrained video editors by training them as image editors on single images, enabling new video editing tasks and state-of-the-art results on OpenVE-Bench and OpenVE++.

### 7. PyRUA-Dex: Learning Programmatic Policies for Dexterous Bimanual Manipulation with Coding Agents

[Read Paper](https://huggingface.co/papers/2611.33144)

A framework enables coding agents to develop dexterous manipulation policies without demonstrations by exploring and distilling successful attempts into a program, then refining it for strong simulated and real-world results.

### 8. DocSEAM: Bidirectional Cross-Page Encoding for Multi-Page Document Understanding

[Read Paper](https://huggingface.co/papers/2611.34420)

DocSEAM is a lightweight module between a pretrained vision encoder and language model that enriches page visual tokens with neighboring-page context via center-anchored masked self-attention, improving multi-page document VQA and reducing boundary sensitivity.

### 9. HOG-WAM: Improving Visual Robustness through Future Gradient Prediction

[Read Paper](https://huggingface.co/papers/2611.34811)

HOG-WAM improves visually robust control in world-action models by adding future Histograms of Oriented Gradients as auxiliary structural supervision through a training-only HOG diffusion transformer, yielding higher success under observation corruptions while preserving the original RGB-to-action inference architecture.

### 10. EgoUseBench: Benchmarking Computer-Use Agents under Evolving First-Person Contexts

[Read Paper](https://huggingface.co/papers/2611.34466)

AI wearables, such as smart glasses, extend the context available to agents beyond digital environments to users' physical surroundings. With these devices, computer tasks can originate from users' ongoing activities rather than solely from explicit instructions. Through first-person observations, agents can identify objects and activities relevant to user requests and ground digital tasks in the surrounding context. We call this interaction paradigm Ego Use. However, first-person observations are limited by the wearer's viewpoint and may provide incomplete or ambiguous evidence of ongoing activities. As new observations arrive, agents may need to revise their understanding of the physical context and user intent, and reassess whether prior digital actions remain valid. Whether current agents can reliably handle these changes remains insufficiently evaluated. We introduce EgoUseBench, a benchmark for evaluating Ego Use across three interaction settings with different levels of first-person context. Computer-Use relies on explicit instructions, Offline Ego Use provides prerecorded first-person context before execution, and Online Ego Use provides continuous observations as user intent evolves. Based on over 74 hours of real-world first-person recordings, we construct 693 tasks across the three interaction settings. To capture task evolution, we track first-person observations, user requirements, and website state separately. We further group tasks by how information changes during execution, resulting in three types: Graft (revised user requirements), Refresh (new first-person observations), and Clarify (agent-initiated information seeking). We evaluate advanced models and find that average success rates in Online Ego Use are 26.3% lower than in Computer-Use. This result reveals the difficulty of keeping task execution aligned with evolving first-person context and user intent.
