---
layout: post
title: "Daily AI Research Papers - Wednesday, November 19, 2025"
date: 2025-11-19
---

Keywords: large language models, Reinforcement fine-tuning, supervised fine-tuning, reinforcement learning, large video language models, VideoP2R, perception, reasoning

---

### 1. VIDEOP2R: Video Understanding from Perception to Reasoning

[Read Paper](https://huggingface.co/papers/2511.11113)

VideoP2R, a process-aware reinforcement fine-tuning framework, improves video reasoning and understanding by modeling perception and reasoning separately, achieving state-of-the-art results on multiple benchmarks.

### 2. Think-at-Hard: Selective Latent Iterations to Improve Reasoning Language Models

[Read Paper](https://huggingface.co/papers/2511.08577)

Think-at-Hard (TaH) dynamically refines only hard tokens in LLMs using a neural decider and LoRA, improving reasoning performance with minimal additional parameters or iterations.

### 3. AraLingBench A Human-Annotated Benchmark for Evaluating Arabic Linguistic Capabilities of Large Language Models

[Read Paper](https://huggingface.co/papers/2511.14295)

AraLingBench evaluates Arabic and bilingual LLMs' linguistic competence using a benchmark with expert-designed questions across grammar, morphology, spelling, reading comprehension, and syntax, revealing gaps between surface proficiency and true comprehension.

### 4. A Style is Worth One Code: Unlocking Code-to-Style Image Generation with Discrete Style Space

[Read Paper](https://huggingface.co/papers/2511.10555)

A novel method, CoTyle, generates images in consistent visual styles using unique numerical style codes, filling an academic gap in code-to-style image generation.

### 5. Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework

[Read Paper](https://huggingface.co/papers/2511.13189)

Foundation models have revolutionized artificial intelligence across numerous domains, yet their transformative potential remains largely untapped in Extreme Multi-label Classification (XMC). Queries in XMC are associated with relevant labels from extremely large label spaces, where it is critical to strike a balance between efficiency and performance. Therefore, many recent approaches efficiently pose XMC as a maximum inner product search between embeddings learned from small encoder-only transformer architectures. In this paper, we address two important aspects in XMC: how to effectively harness larger decoder-only models, and how to exploit visual information while maintaining computational efficiency. We demonstrate that both play a critical role in XMC separately and can be combined for improved performance. We show that a few billion-size decoder can deliver substantial improvements while keeping computational overhead manageable. Furthermore, our Vision-enhanced eXtreme Multi-label Learning framework (ViXML) efficiently integrates foundation vision models by pooling a single embedding per image. This limits computational growth while unlocking multi-modal capabilities. Remarkably, ViXML with small encoders outperforms text-only decoder in most cases, showing that an image is worth billions of parameters. Finally, we present an extension of existing text-only datasets to exploit visual metadata and make them available for future benchmarking. Comprehensive experiments across four public text-only datasets and their corresponding image enhanced versions validate our proposals' effectiveness, surpassing previous state-of-the-art by up to +8.21\% in P@1 on the largest dataset. ViXML's code is available at https://github.com/DiegoOrtego/vixml.

### 6. Can World Simulators Reason? Gen-ViRe: A Generative Visual Reasoning Benchmark

[Read Paper](https://huggingface.co/papers/2511.13853)

Gen-ViRe benchmarks video models on reasoning abilities using a framework that decomposes Chain-of-Frames reasoning into cognitive dimensions and subtasks.

### 7. Agent READMEs: An Empirical Study of Context Files for Agentic Coding

[Read Paper](https://huggingface.co/papers/2511.12884)

Agentic coding tools receive goals written in natural language as input, break them down into specific tasks, and write or execute the actual code with minimal human intervention. Central to this process are agent context files ("READMEs for agents") that provide persistent, project-level instructions. In this paper, we conduct the first large-scale empirical study of 2,303 agent context files from 1,925 repositories to characterize their structure, maintenance, and content. We find that these files are not static documentation but complex, difficult-to-read artifacts that evolve like configuration code, maintained through frequent, small additions. Our content analysis of 16 instruction types shows that developers prioritize functional context, such as build and run commands (62.3%), implementation details (69.9%), and architecture (67.7%). We also identify a significant gap: non-functional requirements like security (14.5%) and performance (14.5%) are rarely specified. These findings indicate that while developers use context files to make agents functional, they provide few guardrails to ensure that agent-written code is secure or performant, highlighting the need for improved tooling and practices.

### 8. REVISOR: Beyond Textual Reflection, Towards Multimodal Introspective Reasoning in Long-Form Video Understanding

[Read Paper](https://huggingface.co/papers/2511.13026)

Self-reflection mechanisms that rely on purely text-based rethinking processes perform well in most multimodal tasks. However, when directly applied to long-form video understanding scenarios, they exhibit clear limitations. The fundamental reasons for this lie in two points: (1)long-form video understanding involves richer and more dynamic visual input, meaning rethinking only the text information is insufficient and necessitates a further rethinking process specifically targeting visual information; (2) purely text-based reflection mechanisms lack cross-modal interaction capabilities, preventing them from fully integrating visual information during reflection. Motivated by these insights, we propose REVISOR (REflective VIsual Segment Oriented Reasoning), a novel framework for tool-augmented multimodal reflection. REVISOR enables MLLMs to collaboratively construct introspective reflection processes across textual and visual modalities, significantly enhancing their reasoning capability for long-form video understanding. To ensure that REVISOR can learn to accurately review video segments highly relevant to the question during reinforcement learning, we designed the Dual Attribution Decoupled Reward (DADR) mechanism. Integrated into the GRPO training strategy, this mechanism enforces causal alignment between the model's reasoning and the selected video evidence. Notably, the REVISOR framework significantly enhances long-form video understanding capability of MLLMs without requiring supplementary supervised fine-tuning or external models, achieving impressive results on four benchmarks including VideoMME, LongVideoBench, MLVU, and LVBench.

### 9. MVI-Bench: A Comprehensive Benchmark for Evaluating Robustness to Misleading Visual Inputs in LVLMs

[Read Paper](https://huggingface.co/papers/2511.14159)

MVI-Bench evaluates the robustness of Large Vision-Language Models against misleading visual inputs using a hierarchical taxonomy and a novel sensitivity metric, revealing significant vulnerabilities.

### 10. Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2511.14460)

A new training framework for RL-based LLM Agents is introduced, extending MDP methodology and demonstrating effectiveness on Multihop QA tasks.
