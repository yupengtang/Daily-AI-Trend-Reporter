---
layout: post
title: "Daily AI Research Papers - Thursday, November 12, 2026"
date: 2026-11-12
---

Keywords: language models, fine-tuning, curriculum learning, scaling patterns, instruction-data baselines, mathematical reasoning, low-bit quantization, LLM agents

---

### 1. Scaling Mathematical Proof Supervision from Structured Textbooks

[Read Paper](https://huggingface.co/papers/2610.40394)

A large-scale corpus of structured proof-supervision records derived from university mathematics textbooks improves language model mathematical reasoning by 16.7 percentage points over a base model when used for fine-tuning.

### 2. SteerDistill: Quantizing LLM Agents via Closed-Loop Teacher Steering

[Read Paper](https://huggingface.co/papers/2611.13410)

Low-bit quantized LLM agents can recover much of their task success through closed-loop guidance from a full-precision teacher, followed by unassisted student training on corrected and reference turns.

### 3. Incentivizing Self-Learning in Large Language Models via Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2610.39671)

Large language models (LLMs) draw on parametric knowledge and retrieval-augmented generation (RAG) to solve knowledge-intensive tasks. Yet problem solving in unfamiliar domains calls for tightly coupled knowledge acquisition and reasoning, beyond factual retrieval and aggregation or learning solely from a preassembled context. We study self-learning -- autonomously retrieving, studying, and applying knowledge from long-form reference materials on the fly. To investigate this capability, we formulate a textbook-based mathematical problem-solving task. LLMs must navigate textbook exposition, learn and apply relevant principles, and construct proofs for exercises, with reference solutions withheld. We introduce a BookNavigator interface to support this human-inspired study process and incentivize self-learning through reinforcement learning (RL) with fine-grained rewards. Experiments show that our method improves performance across in-domain and out-of-domain textbooks and subjects, raising Intern-S2-Preview-35B's overall proof accuracy from 51.5% to 79.5% and outperforming RL-trained textbook-free and fixed-retrieval baselines. These gains extend to complex long-context benchmarks involving different task types, with improvements over the base model of 1.7 and 2.3 percentage points on BrowseComp-Plus and CL-bench, respectively. Together, these findings suggest that RL can foster a domain-agnostic self-learning strategy, highlighting the potential of self-learning from long-form corpora to strengthen broader problem-solving capabilities on complex, long-context tasks.

### 4. Read Less, Solve More: Token-Efficient Sparse Reading of External Content for AI Agents

[Read Paper](https://huggingface.co/papers/2608.22237)

Web-facing AI agents repeatedly access webpages, documents, repositories, and tool responses, yet reading interfaces often expose entire objects when only sparse evidence is needed. This over-reading increases token and latency costs and can dilute relevant evidence. We present SparseRead, a reusable reading control layer that governs content admission without modifying the agent model. Its Read Intent Protocol tracks evolving needs and source coverage; Adaptive Reader Backends select how evidence is located and exposed; and the Budgeted Read Gate bounds cumulative raw exposure per episode with faithful full-view fallback. A unified Reader interface supports interchangeable evidence acquisition across heterogeneous resources. Across six models, including Claude Opus 5, and five scenarios, SparseRead reduces total tokens by up to 92.9% and wall time by up to 89.0%, preserving or improving task quality in 97% settings. On Web deep-search tasks, it reduces tokens by 31.0--43.0% and task completion time by 23.0--37.0%. Ablations validate the three designs, while integration with three agent frameworks demonstrates portability.

### 5. Ensemble Routing for Prompt-Adaptive Cost-Aware LLM Inference

[Read Paper](https://huggingface.co/papers/2611.10193)

PISA and PISA-E use prompt-adaptive capability estimation to route prompts or select cost-aware ensembles, improving cost-performance trade-offs across LLM routing benchmarks.

### 6. GTAGym: Real AAA Game Tasks Environment for Evaluation and Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2611.13028)

AAA games offer a rich testbed for evaluating agent perception, reasoning, and control. However, evaluating whether agents can complete tasks in AAA games is challenging and often human-annotated because task outcomes are often difficult to verify automatically, let alone applying it for Reinforcement Learning (RL). We introduce GTAGym, an environment and framework for evaluation and RL on GTA V tasks. Its evaluation benchmark, GTAVBench, derived from GTA V main-story missions, contains 103 online tasks reconstructed across nine task families, alongside 47 image-based shooting and sniping tasks using GTA V imagery. A web server provides access to the online environment and evaluation interface, supporting parallel testing through screenshots and keyboard-mouse actions with automatic engine-state outcome checks. We evaluate six frontier LLMs on GTAVBench with textual task instructions and on its GTAVBench-AGI branch, where instructions appear only in the first screenshot and a harness automatically compresses interaction history. The best configuration achieves 70.9% overall accuracy in GTAVBench-AGI, yet aerial navigation, evasion and interception remain difficult. GTAGym also provides a scalable generator that samples task parameters within user-specified ranges and option sets under fixed templates, creating batches of 3D and image-based shooting environments for RL with automatic reward and termination signals. LoRA RL of Qwen3.8-27B on generated on-foot navigation tasks improves the in-domain performance and transfers to out-of-domain task categories, and native GTA V tasks from 15% to 23%.

### 7. Evolving Infra: Agent-Adaptive LLM Serving at Scale

[Read Paper](https://huggingface.co/papers/2611.12304)

Agentic workloads are a major source of LLM inference demand. Existing prefill--decode (PD) disaggregated serving systems offer limited flexibility for runtime adaptation of compute allocation and KV cache management, and rely on coarse signals that only partly capture the diverse and dynamic agent workload. These limitations lead to load imbalance between prefill and decode nodes and eviction of KV caches that agents will reuse. We present Evolving Infra, an agent-adaptive serving framework that adapts compute allocation to agent workloads and KV cache management to agent context management at runtime. We decouple compute allocation from key--value (KV) cache management, allowing nodes to switch efficiently between prefill and decode. Building on this design, an online planner matches the PD split to workload composition, while an agent-aware distributed KV manager guides cache eviction and transfer with signals from agent context management. Across coding, math, and computer-use tasks with GLM-5.2 and Kimi K2.6 on up to 256 GB300 GPUs, Evolving Infra reduces mean end-to-end request latency and P95 TTFT by up to 4.1x and 14x over Dynamo, NVIDIA's production framework.

### 8. Where Does Alignment Emerge and What Does It Mean? A Closer Look at the Platonic Representation Hypothesis

[Read Paper](https://huggingface.co/papers/2611.12869)

Cross-modal representational alignment between unimodal language and vision models increases with model capability when captions are clean and informative, but remains below same-modality alignment and can exceed explicitly aligned vision-language models.

### 9. Interactive Visual Gym: Evaluating Visual Interaction Models with Simulated Users in Interactive Worlds

[Read Paper](https://huggingface.co/papers/2611.08286)

Interactive Visual Gym enables closed-loop evaluation and training of visual interaction models by jointly simulating task environments and human users, exposing limitations in state-of-the-art VLMs and improving performance through closed-loop training.

### 10. FAME-QP: A GPU-Accelerated Quadratic-Programming Solver for Real-Time Model Predictive Control

[Read Paper](https://huggingface.co/papers/2611.13766)

We present FAME-QP, a GPU-accelerated solver for dense quadratic programs arising in linear model predictive control. FAME-QP combines constraint-dependent operator representations with fused single- and two-kernel iterations replayed through CUDA graphs, automatically selecting the execution configuration through device calibration and problem-specific timing. It probes a cached operator bank to select the initial penalty and updates operators lazily when problem matrices change. Numerical benchmarks and simulated humanoid and quadrotor control experiments demonstrate reduced latency relative to competing GPU solvers while maintaining solution accuracy and control performance.
