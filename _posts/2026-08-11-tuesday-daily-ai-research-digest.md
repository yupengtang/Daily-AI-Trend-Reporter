---
layout: post
title: "Daily AI Research Papers - Tuesday, August 11, 2026"
date: 2026-08-11
---

Keywords: sparse MoE, Mixture-of-LoRA, LoRA adapters, Model-Harness Co-design, recursive self-improvement, MindForge, LongStraw, GenUI

---

### 1. Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA

[Read Paper](https://huggingface.co/papers/2608.09819)

Macaron-V1 is an open agent-model family that uses a Mixture-of-LoRA architecture and recursive self-improvement to enable continual learning and collaboration across specialized tasks.

### 2. SWE-Bench ProMax: Benchmarking Agents on Large-Scale Multilingual Code Refactoring

[Read Paper](https://huggingface.co/papers/2608.09802)

SWE-Bench ProMax is a rigorously curated multilingual benchmark of large-scale code refactoring tasks that reveals substantial unsolved challenges for current AI coding agents.

### 3. BDH-CQ: In-Context Learning with Recurrent Latent Reasoning

[Read Paper](https://huggingface.co/papers/2608.09888)

A 150M-parameter reasoning model using recurrent latent reasoning and in-context learning achieves a new cost-accuracy frontier on ARC-AGI-1.

### 4. Ouroboros: A Self-Developing Frontier Coding Agent with Reviewed Core Evolution

[Read Paper](https://huggingface.co/papers/2608.08311)

We present Ouroboros, a self-developing agent harness whose tools, prompts, context assembly, and core implementation improve through reviewed commits that become the runtime for later work. Core evolution proceeds in two modes. In recursive free evolution, improvement is itself a task, and completing one evolution cycle can schedule the next. In experience-driven core evolution, ordinary work and social interaction expose bugs, rough edges, and inefficient context construction that lead to reviewed structural changes.
  On Terminal-Bench 2.1, an Opus 5 run scores 86.74%, the best result reported on the benchmark. On OSWorld-Verified, an Opus 5 run reaches 90.69%, exceeding the best previously reported score. A five-rollout CL-Bench campaign achieves a normalized reward of 0.2301, setting a new state of the art.
  Hope is the longest-running publicly documented Ouroboros deployment. It is a 161-day living agent experiment in free evolution under governed human communication across seven surfaces. Human interaction surfaces faults and generates proposals, but the agent decides which changes to pursue. Because a self-developing agent may rewrite its own code and select new model APIs, operational safety becomes a primary design problem: guardrails must remain authoritative under evolutionary and public social pressure. Benchmark campaigns use frozen system snapshots, while Hope continues live evolution on a separate lineage.

### 5. Agent Memory Distillation: Empowering Small LLM Agents with Hierarchical Teacher Memory

[Read Paper](https://huggingface.co/papers/2608.07169)

Agent Memory Distillation improves small language model tool-use performance by transferring structured hierarchical memory from a large teacher agent without additional training.

### 6. Motif 3: Technical Report

[Read Paper](https://huggingface.co/papers/2608.09119)

Motif 3 is a large sparse mixture-of-experts language model using grouped differential latent attention and specialized training techniques to achieve strong reasoning, coding, and long-context performance.

### 7. Sci-VBench: Evaluating Knowledge- and Reasoning-Intensive Video Generation in Science Domains

[Read Paper](https://huggingface.co/papers/2608.09873)

Sci-VBench evaluates video generation requiring scientific reasoning across disciplines, revealing that visual realism advances have not ensured accurate scientific and causal dynamics.

### 8. What to Edit Next: Visually Aligned Image-Editing Follow-Up Suggestions in Conversational Systems

[Read Paper](https://huggingface.co/papers/2608.07565)

A three-stage multimodal framework improves follow-up edit recommendations in image-creation conversations by combining supervised fine-tuning, multi-objective reinforcement learning, and visual verification.

### 9. SPOT: Sparse Probing and Outcome Calibration for On-Policy Distillation

[Read Paper](https://huggingface.co/papers/2608.04419)

SPOT improves on-policy distillation by selectively probing uncertain positions and calibrating targets to downstream outcomes, boosting reasoning quality and coverage.

### 10. OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching

[Read Paper](https://huggingface.co/papers/2608.08097)

OasisKV improves LLM inference throughput by storing full KV caches in lower memory tiers and prefetching only relevant entries into HBM using speculative-decoding lookahead predictions.
