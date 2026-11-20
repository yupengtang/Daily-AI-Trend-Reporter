---
layout: post
title: "Daily AI Research Papers - Friday, November 20, 2026"
date: 2026-11-20
---

Keywords: video generation, chain-of-thought reasoning, multi-agent workflows, agentic framework, main investigator agent, self-evolving library, forensic inspector sub-agents, visual and audio cues

---

### 1. WeVisForensics: Self-Evolving Agent for Omni AI-Generated Video Detection

[Read Paper](https://huggingface.co/papers/2611.21373)

WeVisForensics coordinates self-evolving visual and audio forensic inspector agents under a dual-timescale reinforcement learning policy to detect generated-video artifacts with localized evidence and high accuracy.

### 2. Mind2Act World: A Diagnostic Benchmark for Embodied Mind, Physical Action, and Mind-Action Composition in Robotic Manipulation

[Read Paper](https://huggingface.co/papers/2611.26444)

A diagnostic benchmark of 180 simulated manipulation configurations evaluates whether embodied systems can combine history-based reasoning, precise execution, and closed-loop environmental feedback, revealing that single-view competence does not transfer to composed tasks.

### 3. WeMM-Video: Pushing the Frontiers of Open-Source Omni-Modal Video Understanding

[Read Paper](https://huggingface.co/papers/2611.26409)

WeMM-Video is a family of Chinese-English bilingual omni-modal foundation models that unify text, image, audio, and video processing for joint audio-visual perception and reasoning, using hybrid thinking modes and large-scale training to achieve strong open-source video understanding results.

### 4. Probing the "Conscious" Mind: Selective Access and Global Workspace in Vision-Language Models

[Read Paper](https://huggingface.co/papers/2611.26590)

Vision-language models (VLMs) integrate visual and textual information, but their outputs do not reveal internal concept representations. Global workspace theory (GWT) proposes that information is selected from parallel brain processes, integrated in a shared workspace, and made available to specialised processes. We introduce the Multimodal Concept Lens (MC-Lens), a multimodal Jacobian lens, to study these functions in VLM concept processing. Its vision encoder and language decoder lenses map intermediate states to a shared semantic space defined by the decoder's output vocabulary to compare concept accessibility across modalities. MC-Lens supports internal concept interventions to examine the role of accessed concepts in downstream computation. (i) Accessibility and selective access. Multimodal concept information is readable in intermediate states and selectively accessible as tasks require. We extend "readout carriers" from one name token to a token set to better capture and express concept information and support counterfactual experiments. (ii) Concepts in downstream computation. Counterfactual experiments show that accessed concept information participates in downstream computation and reasoning. (iii) Development of cross-modal alignment and selective access. We develop two J-space metrics tracking the development of cross-modal concept alignment more clearly than hidden state cosine in controlled VLM training. The experiment also shows that reliable selective access develops later than concept accessibility. (iv) Shared functions across models. Across VLM families and sizes, we find shared patterns of concept accessibility, query-dependent selectivity and cross-task intervention effects despite differences in internal organisation. These findings offer an experimental basis to study GWT in multimodal artificial systems and how it can guide cross-modal alignment improvements.

### 5. Coda-DaVinci: Reimagining Video Generation with Code2Video

[Read Paper](https://huggingface.co/papers/2611.26595)

A task-adaptive agentic framework utilizes 3D proxies, physics simulation, and multi-agent coordination to enhance the structural control, physical plausibility, and visual realism of generated videos.

### 6. Posterior Matching Distillation

[Read Paper](https://huggingface.co/papers/2611.26629)

Posterior Matching Distillation trains a stochastic student to match a teacher diffusion model’s full posterior via tilted mean constraints, enabling high-quality few-step image generation without auxiliary score networks.

### 7. The Mask Is Not the Object: Volumetric Supervision for 3D Gaussian Segmentation

[Read Paper](https://huggingface.co/papers/2611.21383)

A plug-and-play objective, VODA, groups per-pixel contributors by local depth to supervise all Gaussians in selected depth clusters, improving volumetric segmentation and enabling evaluation via Removal-3D and Removal Depth Error.

### 8. Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth

[Read Paper](https://huggingface.co/papers/2611.26603)

A single scalable network estimates dense metric depth for images and video at native resolution, jointly improving metric accuracy, boundary sharpness, and temporal consistency.

### 9. EgoCaptioner: Towards Comprehensive and Fine-Grained Egocentric Video Captioning

[Read Paper](https://huggingface.co/papers/2611.26380)

Egocentric video captioning requires fine-grained detail and accurate timing, yet current multimodal large language models (MLLMs) often miss brief actions, confuse their order, or misinterpret the wearer's motion. Existing caption formats also offer limited support for combining detailed narration with precise timing. We present EgoCaptioner, which pairs a video-level overview of the setting, participants, and the wearer's overall activity with timed steps whose narration adapts to varying perceptual demands and marks key moments at specific timestamps. We develop a coarse-to-fine annotation pipeline that establishes event structure before expanding step descriptions, yielding 50,000 training examples from eight egocentric video sources. Complementary rewards assess factual accuracy and coverage, measure step-level temporal alignment, and penalize fragmented or redundant narration, guiding caption refinement through Group Relative Policy Optimization (GRPO). To improve the reliability of factual supervision, we supplement caption references with pose-derived motion facts when sensor data are available. Across five egocentric benchmarks, EgoCaptioner achieves leading performance among evaluated open-source models in caption quality, caption-conditioned question answering, and temporal alignment. It also surpasses Gemini 3.1 Pro on PhysCapBench and all evaluated proprietary baselines on Minerva-Ego under the caption-to-QA protocol. Ablations support the complementary roles of factual and temporal rewards and show that coarse-to-fine annotation provides more effective supervision than single-pass annotation.

### 10. HiUni: Harmonizing Understanding and Generation in Shared Visual Space

[Read Paper](https://huggingface.co/papers/2611.26107)

Unified multimodal models aim to support visual understanding and generation within a single framework. However, prevailing approaches often separate their visual representations or backbone computation to accommodate the different requirements of these two tasks, which limits the degree of unification. In this paper, we present HiUni, which revisits this design choice and explores whether a pretrained vision language model can be directly extended into a unified model with extensive parameter sharing. Specifically, HiUni reuses the same language-aligned ViT and dense Transformer backbone for both capabilities, while confining task-specific computation to the output side. Yet such a design introduces two key challenges. Generation in pixel space is difficult to optimize, and semantic abstraction in pretrained ViT discards fine spatial details required for synthesis. To address these challenges, HiUni introduces a Pixel Diffusion Head tailored to direct pixel-space optimization and a lightweight HiGate pathway that provides fine-grained spatial details near the output stage. Extensive experiments show that HiUni achieves a similar average understanding score to the pretrained VLM while attaining competitive performance on image generation and editing. Controlled comparisons and analyses further demonstrate that a single dense backbone can effectively support both capabilities without task-specific backbone computation, highlighting a simpler path toward unified multimodal modeling.
