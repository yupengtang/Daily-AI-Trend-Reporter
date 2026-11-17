---
layout: post
title: "Daily AI Research Papers - Tuesday, November 17, 2026"
date: 2026-11-17
---

Keywords: large language models, flow-matching framework, cross-asset flow Transformer, temporal market representations, continuous signed position weights, two-stage training scheme, strategy-supervised imitation, group-relative reinforcement learning

---

### 1. Sage S2.1 Flash Technical Report

[Read Paper](https://huggingface.co/papers/2611.21072)

We introduce Sage S1.5 Flash, a 320-billion-parameter general-purpose mixture-of-experts model designed to reason over scientific evidence and act in interactive environments. Sage Sci-Perception uses dedicated encoders for molecular graphs and biological sequences and represents spectra as ordered patches of measurements. Modality-specific projectors map these representations into the input space of a shared backbone, which processes them alongside text and images. To develop the model's agent capabilities, we construct verifiable environments for tool use, search, and coding, as well as specialized workflows involving machine learning experiments and scientific software. Task completion is checked against execution results, changes in environment state, or submitted artifacts. We evaluate Sage S1.5 Flash on 33 benchmarks across seven domains. In the primary comparison against five baselines, it ranks first on OfficeQA Pro (72.2), MLE-bench Lite (90.3 HumanRank), Harvey LAB, and SpreadsheetBench~2, and second on OSWorld-Verified (85.2), WebArena-Verified (67.6), and DeepResearch Bench~II (54.2). These results show particular strengths in using documentary evidence, producing verifiable workbooks, and carrying out machine learning tasks, alongside competitive performance in desktop and browser workflows that require the model to act on visual feedback.

### 2. K2-Horizon: A Frontier Fleet of Open Source Language Models

[Read Paper](https://huggingface.co/papers/2611.21003)

We present K2-Horizon, a fleet of six fully open language models, from a 0.9B dense model to a 375B-A23B MoE, built for general-purpose reasoning and agentic work across math and science, software engineering, deep research, office automation, and beyond. The models lead or are competitive with strong open models of their size on agentic and coding benchmarks. On the modeling side, to strengthen reasoning, we introduce Mixture-of-Value Attention (MoVA), applying MoE to attention value projections, and softplus attention gating, a more flexible, theoretically motivated gate. Our 36B-A4B model uses both and nearly matches a 32B dense model trained on the same data, with one-seventh the active parameters. We also introduce Uno, lossless diffusion adapters that speed up decoding by up to 2.2x without a separate draft model. Our release follows the LLM360 definition of open source for transparency and reproducibility: the full development history, including intermediate checkpoints from every training stage, pre-training data, training logs, and all training, evaluation, and infrastructure code, notably our production-scale xLLM framework for dense and MoE training and the RL360 reinforcement-learning stack. Beyond the data itself, we release and document in detail every pipeline that built it, the part of model development that open releases most rarely disclose. These pipelines produced roughly 10T synthetic pre-training tokens, showing that synthetic data can be a major ingredient of pre-training at scale. Four of the models share one 22T-token pre-training corpus and recipe, revealing consistent scaling behavior and enabling controlled studies of scale and sparsity. All artifacts are available at https://ifm.ai/k2/

### 3. AllocFlow: End-to-End Portfolio Trading Action Generation from Raw Market Data

[Read Paper](https://huggingface.co/papers/2611.19937)

An end-to-end flow-matching framework generates joint portfolio allocations from raw multi-asset market histories via a cross-asset flow Transformer and a two-stage imitation and group-relative reinforcement learning procedure, improving risk-adjusted returns on cryptocurrency perpetual futures.

### 4. InclusionMed: Benchmarking AI for Inclusive Healthcare

[Read Paper](https://huggingface.co/papers/2611.21218)

InclusionMed is an open medical AI evaluation initiative that audits and reproduces benchmarks across healthcare roles and capabilities, revealing that model rankings and inference costs vary by context and motivating broader, context-specific evaluation.

### 5. SPILLWAY: Separating Target Admission from Sanitization to Defend Tool-Using Agents against Prompt Injection

[Read Paper](https://huggingface.co/papers/2611.07222)

Tool-using agents act on email, files, and payments, so an instruction hidden in a web page or a shared document can make an agent send data or money to an attacker. Many runtime defenses put a language model on the decision that lets an action through, so the decision depends on a model that may read the attacker's text and adds model calls to every step. Provenance defenses keep the model off the final check but need a model to plan or route each task first. Where an action goes can instead be read from the tools' typed returns: an email's sender or a payment's account arrives in a typed field, while injected instructions arrive as free text. SPILLWAY turns this observation into a deterministic floor with no per-task plan. The check admits a call when each target value that appears in tool output but not in the user's task was returned by a read-only tool in a typed field or is covered by a fixed delegation rule. A deterministic sanitizer removes instruction-like regions its fixed rules match before the model reads them and retains the removed spans as evidence, and a local judge may restore a refused call, never one whose target lies in the removed text. Across three backbones, and seven task suites on one of them, no standard templated injection redirects an action to an attacker's target under SPILLWAY, at the highest observed mean utility and composite scores (NRP, ERS) of the defenses we run under that attack. A forged chat template also produces a raw zero, and mean in-scope attack success under a released reinforcement-learning attacker is .007. SPILLWAY reaches this floor with no defense-triggered backbone re-run and almost no added model call. As personal AI agents act on users' email, files, and payments, SPILLWAY gives them an auditable refusal floor on where their actions go that keeps them useful and holds across backbones.

### 6. SWE-CoWork: Benchmarking Coding Agents as Digital Employees in Organizations

[Read Paper](https://huggingface.co/papers/2611.20218)

A benchmark of realistic software engineering assignments reveals that current coding agents struggle to track requirements from clarification through implementation, verification, and handoff, completing few tasks despite partial test success.

### 7. ScaleVLA: Scaling Vision-Language-Action Models to 552 Billion Parameters

[Read Paper](https://huggingface.co/papers/2611.21607)

Frontier Vision-Language Models (VLMs) such as GPT-6 Astra have demonstrated remarkable capabilities in robot control, suggesting that broad pretraining provides knowledge valuable for physical action. Yet most Vision-Language-Action (VLA) policies still rely on much smaller VLM backbones, leaving this potential largely untapped. We ask whether frontier-scale representations can provide a stronger foundation for robot learning. We introduce ScaleVLA, which connects a 552B DeepSeek-V4.1-Flash backbone to a trainable action DiT, and systematically compare VLMs from 2B to 552B under a shared training setup. We evaluate three fundamental capabilities of robot foundation models. In sample efficiency, just 10 demonstrations outperform a 2B backbone trained with 50 (87.8% vs. 85.6%). In generalization, LIBERO-PRO success reaches 45.1%, compared with at most 31.3% for smaller backbones. In precise manipulation, real-world charger insertion improves from 4/10 to 9/10 successes, with gains concentrated in contact-sensitive alignment. Despite its scale, ScaleVLA achieves 216 ms median latency per action chunk. Our findings suggest that frontier VLMs can serve not only as reasoning agents, but also as powerful representational foundations for efficient and generalizable robot policies.

### 8. PaperCritic: Training AI Reviewers to Assess the Soundness of Papers

[Read Paper](https://huggingface.co/papers/2611.17834)

An AI reviewer trained by reinforcement learning to jointly maximize precision and recall of soundness review points outperforms frontier models and other training methods and transfers to other review domains.

### 9. VoxMel TTS Technical Report

[Read Paper](https://huggingface.co/papers/2611.20254)

VoxMel is a tokenizer-free autoregressive TTS framework that uses an invertible two-axis Haar transform to decompose mel spectrograms into Coarse-Mel and residual tiers, enabling stable high-fidelity generation with strong recognition and speaker-similarity results.

### 10. Accelerate Bench: Evaluating Agents for Application-Level Program Optimization

[Read Paper](https://huggingface.co/papers/2611.20882)

Application optimization requires coding agents to discover useful changes and coordinate them across a complete program while preserving the required computation. We introduce Accelerate Bench, a benchmark of 32 CPU optimization tasks adapted from scientific applications and supercomputing competitions. Each task provides a working program and specifies the workload, correctness requirements, timing boundaries, and permitted edits, leaving agents to choose where and how to optimize. Public testing supports iterative development, and a separate verifier scores the final submission. On benchmark version 1.1, we evaluate 16 coding agents, each pairing a model with a harness. Their Score@1 ranges from 25.5% to 67.4%. Most of their programs are correct, and the size of the speedup separates them: the two leading agents are at least 2x faster on 22 or more of the 32 tasks, while the other fourteen reach 2x on 8 to 17 and leave 125 of their 412 correct outcomes below 1.25x. Controlled replays show why coordination across an application matters: identical source code runs between 0.29x and 2.12x depending only on build and runtime settings, and one code change is worth 1.01x or 6.4x depending on another. Accelerate Bench pairs executable workloads with agents' programs and trajectories to evaluate both their gains and the process behind them.
