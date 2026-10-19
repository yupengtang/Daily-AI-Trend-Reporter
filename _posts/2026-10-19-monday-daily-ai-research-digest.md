---
layout: post
title: "Daily AI Research Papers - Monday, October 19, 2026"
date: 2026-10-19
---

Keywords: in-context learning, System 2 reasoner, System 1 controller, close-loop coordination, embodied control, long-horizon preservation, physical-state verification, vision language models

---

### 1. Has GPT-6-Astra Achieved Embodied Intelligence?

[Read Paper](https://huggingface.co/papers/2610.22471)

GPT-6-Astra achieves high performance across diverse robotic tasks via in-context learning, yet real-world deployment reveals gaps in demonstration cost, execution time, and practical embodied intelligence.

### 2. HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents

[Read Paper](https://huggingface.co/papers/2610.23479)

HexaShell is a coding agent that controls a dual-arm robot via vision and self-written tools, achieving state-of-the-art results on RoboDojo but struggling with contact-heavy tasks.

### 3. BizBench: Unveiling Hidden Challenges in Deploying Long-Horizon Agents for Real-World Business

[Read Paper](https://huggingface.co/papers/2610.20424)

BizBench evaluates long-horizon business agents on intermediate execution validity using real-world payment procedures, revealing that frontier models frequently violate constraints despite achieving partial goal success.

### 4. Learning to Think and Explore in 3D World

[Read Paper](https://huggingface.co/papers/2610.21783)

ACT3D trains vision-language models via reinforcement learning to actively explore 3D environments from single first-person views, improving answer accuracy through adaptive, question-conditioned navigation without direct exploration supervision.

### 5. Into the Unknown: Benchmarking AI agents on open scientific research problems

[Read Paper](https://huggingface.co/papers/2610.19159)

Researchers benchmarked AI agents on open scientific problems by having them draft follow-up papers from unpublished directions, finding that none met expert standards and most contained fabricated claims or contradictory evidence.

### 6. LiveMACE: Process-Aware Evaluation of LLM Agent Capabilities in Evolving Markets

[Read Paper](https://huggingface.co/papers/2610.09872)

LiveMACEBench evaluates persistent LLM agents in live financial markets using decision traces to reveal gaps between outcomes and underlying capabilities.

### 7. Pack Light, CarryOn: Building Object-Oriented Context for Long-Horizon Agents

[Read Paper](https://huggingface.co/papers/2610.18784)

CarryOn organizes long-horizon agent context around reusable task objects rather than interaction histories, improving efficiency and performance on complex benchmarks.

### 8. Speculative Sampling in Interactive Video Generation

[Read Paper](https://huggingface.co/papers/2610.20556)

Speculative Sampling enables interactive video generation with large bidirectional windows by caching predictions and delivering them incrementally, reducing latency without sacrificing quality.

### 9. Scene-Designer: Learning Design Intelligence for Agentic 3D Scene Generation

[Read Paper](https://huggingface.co/papers/2610.23121)

Generating complete, coherent, and usable 3D indoor scenes from natural language requires strong design intelligence: the ability to design high-quality scenes and critique their quality. However, existing approaches either rely on general-purpose models lacking domain-specific design experience or learn from static scenes. In this work, we introduce Scene-Designer, an agentic framework that learns design intelligence through complementary design and critique experiences. We construct two datasets: Scene-OTA-35K, which provides Observation--Thought--Action supervision for learning scene design and efficient error correction, and Scene-OTC-15K, which provides Observation--Thought--Critic supervision for learning scene quality critique across physical validity, functional correctness, and layout coherence. Based on these experiences, we propose Scene-GRPO, a Critic-guided GRPO framework that improves design decisions and reduces unnecessary trial-and-error corrections. Extensive experiments demonstrate that Scene-Designer achieves superior performance in physical validity, functional correctness, and layout coherence, while the learned Scene-Critic significantly outperforms general-purpose models in scene critique. Moreover, test-time Critic guidance further improves scene generation, showing that the Critic can provide useful guidance beyond training. Additional analysis suggests a trend of mutual enhancement between scene design and critique abilities.

### 10. OpenMobile-2: Building Versatile Mobile Agents with Scalable Environments and App-Native Tools

[Read Paper](https://huggingface.co/papers/2610.14157)

OpenMobile-2 is an open mobile agent trained with diverse simulated apps, large-scale supervised and reinforcement learning data, and hybrid GUI-tool use, achieving strong benchmark results and real-device transfer.
