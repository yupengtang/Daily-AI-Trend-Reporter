---
layout: post
title: "Daily AI Research Papers - Tuesday, November 10, 2026"
date: 2026-11-10
---

Keywords: on-policy distillation, tactile foundation model, unified representation, heterogeneous tactile data, tactile tokens, plug-and-play, policy models, diffusion policy

---

### 1. Learning Heterogeneous Tactile Representation for General Dexterous Manipulation

[Read Paper](https://huggingface.co/papers/2611.12806)

LingBot-Touch is a tactile foundation model that learns a unified, extensible representation across heterogeneous tactile sensors, yielding plug-and-play tactile tokens that consistently enhance manipulation policies such as diffusion policy and VLA models.

### 2. WorldFoundry: Towards Unified Infrastructure for World Intelligence

[Read Paper](https://huggingface.co/papers/2611.10771)

WorldFoundry offers open-source infrastructure for reproducible inference and evaluation of generative world models across video, 3D/4D, embodied action, and interactive systems, with traceable evidence and benchmarks.

### 3. Sage-Taiyi: A Metacognitive Architecture for Scientific Discovery

[Read Paper](https://huggingface.co/papers/2611.12762)

We introduce Sage-Taiyi, a metacognition-inspired architecture that uses accumulated scientific understanding to guide what to investigate next and how to investigate it. Through explicit monitoring and control, the system revises its understanding of evidence, assumptions, and unresolved questions, then uses these assessments to adapt research targets and select reusable reasoning methods. On Taiyi-Bench, an AI co-scientist ideation benchmark comprising 265 tasks from 50 papers across ten disciplines, Sage-Taiyi ranks first on all eight metrics of historical rediscovery and open-ended ideation, achieving an Overall score of 54.0 compared with 47.9 for the strongest baseline, AI Scientist-v2. It also performs well on two external scientific ideation benchmarks. Five expert-guided collaborations further connect its proposals to experimental and computational validation across chemistry, materials science, and engineering. In chemistry, reaction-condition optimization raises the yield of cobalt-catalyzed carbonylation from trace amounts to 84% and photocatalytic allylic amination from 13% to 69%. These results support metacognitive regulation as an approach to scientific ideation and its progression toward practical validation.

### 4. Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning

[Read Paper](https://huggingface.co/papers/2611.12886)

Reframe is a two-stage post-training framework that aligns visual agents’ observation states through evidence-aware reinforcement learning and observation-gated on-policy distillation, improving tool use and outperforming Mix-RL and vanilla MOPD across image and video benchmarks.

### 5. Flide: Accelerating Decision Inference with Exact State Forking

[Read Paper](https://huggingface.co/papers/2610.42148)

Flide is an exact inference engine that accelerates decision-model LLM agents by decoupling logical request trees from physical execution and using ForkAttention to preserve exact branch evaluations.

### 6. ReproBridge: Building Reproducibility into ML Research Artifacts

[Read Paper](https://huggingface.co/papers/2611.08452)

Reproducing machine learning experiments requires reconstructing procedures from information scattered across papers, released code, and supporting artifacts. For coding agents, reproduction success can therefore depend on their ability to fill in missing details, rather than on the released artifacts alone. We ask whether preserving recovered procedures and decisions improves subsequent independent reproduction. We define a standardized artifact design comprising a refined codebase, environment setup, experiment launchers, and a reproduction guide, and introduce ReproBridge, an agent workflow for constructing these refined reproducible artifacts. ReproBridge-Bench provides 1,029 tasks from 387 experiments across 36 papers, anchored by five human-verified reference scores per task. Agents attempt tasks in fresh sessions with raw code or refined artifacts; a separate Auditor checks execution evidence. Across four evaluated models, refined artifacts raise mean reproduction success from 76.4% to 92.6% and reduce downstream API spending by 19.7% on average. The cross-model success-rate spread narrows from 21.6 to 3.9 percentage points, with all four models achieving mean success rates above 90%. Removing any component reduces success in the ablation study, with the largest losses from code and environment removal. These results suggest that preserving execution knowledge helps even models with lower raw-code success reproduce experiments more reliably and at lower API cost.

### 7. Longtail-SWE: High-Quality SWE Tasks for Underrepresented Languages

[Read Paper](https://huggingface.co/papers/2611.10304)

Challenging repository-level benchmarks remain limited for many programming language ecosystems, where filtering scarce candidates with ambiguous descriptions or defective tests can discard valuable engineering problems. We introduce Longtail-SWE, a benchmark of 202 issue-resolution tasks across 13 languages with limited coverage in existing repository-level benchmarks: Clojure, Dart, Elixir, Erlang, F#, Haskell, Julia, Kotlin, Lua, OCaml, R, Scala, and Swift. Its construction uses a solver--judge--repairer pipeline: a solver attempts each candidate task, and a judge reviews the task together with the solver's trajectory and test results to guide revisions to tests and task descriptions. These revisions aim to correct tests that reject valid alternative implementations or accept incorrect solutions while retaining challenging tasks and preserving their intended requirements. We evaluate nine model configurations, including frontier models, and analyze task-solving performance, consistency across repeated attempts, and resource usage. We release all benchmark instances, executable environments, complete construction and evaluation tooling, and agent trajectories, together with an additional set of easier tasks to support evaluation of smaller models.

### 8. InfraRSI: Harnessing Recursive Self-Improvement for Training Infrastructure

[Read Paper](https://huggingface.co/papers/2611.11558)

While Recursive Self-Improvement (RSI) accelerates isolated algorithm and kernel synthesis, distributed LLM training depends on tightly coupled launch settings, operator kernels, and communication schedules, where single edits often hit out-of-memory (OOM) boundaries, compiler conflicts, measurement jitter, or silent numerical drift. We present InfraRSI, the agentic harness that turns open-ended human commands into verified training optimizations via a five-stage workflow: init locks editable scope and a fixed task contract over workload, loss, and timing invariants to prevent reward hacking, doctor profiles baseline phases and calibrates cluster noise, run co-optimizes launch knobs and code patches before re-verifying top candidates against the baseline and logging findings to a cross-hardware experience bank, and report and apply audit source paths, track failed-trial costs, and commit promoted changes. Rather than stepping greedily from a single incumbent, InfraRSI initializes from diverse experience-seeded configurations and fits global execution-time and censored peak-memory models across all trials, including OOMs. Using these models, it prices per-setting memory costs, revises miscalibrated LLM priors from live runs, and pairs memory-hungry optimizations with memory savers and verified code patches. Across four single- and multi-node dense and Mixture-of-Experts (MoE) LLM workloads, InfraRSI achieves a speedup of up to 2.75times and lifts mean Model FLOPs Utilization (MFU) from 17.4% to 30.0%, outperforming existing tuners and coding agents while transferring experience across GPUs. As it improves training infrastructure efficiency, InfraRSI forms a two-level Recursive Self-Improvement loop---revising priors online within runs and distilling experience across clusters---to push foundation model scaling.

### 9. SIRIS: Can LLMs Tell When Simulation Errors Matter in AI-Driven Systems Research

[Read Paper](https://huggingface.co/papers/2611.10329)

A benchmark probes whether large language models can determine when imperfect simulator limitations invalidate recommended system designs, finding that identifying a limitation does not ensure correct design choices and that review-based self-correction yields mixed results.

### 10. How Much Training-Inference Mismatch Can On-Policy Distillation Tolerate?

[Read Paper](https://huggingface.co/papers/2611.11450)

On-policy distillation tolerates substantial training-inference mismatch from quantized inference and asynchronous rollouts, with model- and layer-specific limits, and a lightweight admission check plus FP8 recipes can exploit this tolerance to improve training efficiency.
