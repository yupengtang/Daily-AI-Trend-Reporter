---
layout: post
title: "Daily AI Research Papers - Friday, November 27, 2026"
date: 2026-11-27
---

Keywords: computer-using agents, generalist agent, long-horizon workflows, cross-platform benchmark, execution environment, platform-specific adapters, trajectory logging, executable final-state evaluators

---

### 1. Stable Gated DeltaNet-2: Understanding and Mitigating Recurrent State Instability

[Read Paper](https://huggingface.co/papers/2611.34051)

Linear attention offers an efficient solution for long-sequence modeling via linear complexity and fixed-size recurrent states. Gated DeltaNet-2 (GDN-2) introduces independent channel-wise control over erasure and writing, enhancing state-update expressiveness and in-context retrieval, yet it suffers from training failures during long-context training. We trace these failures to the asymmetry between reading and modification directions during erasure, which yields non-normal erase operators whose directional gains accumulate across tokens, triggering exponential state growth even without new writes. To resolve this, we propose , employing symmetric erasure to ensure non-expansive state propagation and decay-regulated residual writing to preserve writing flexibility. Theoretically, we establish an O(T) upper bound on recurrent state growth while retaining chunkwise parallel computation. Experiments on ~1.6B parameter models show that  achieves stable training alongside improved long-context retrieval. Compared to GDN-2, it boosts the 128K average RULER score by 6.80 points in the recurrent setting and average accuracy across nine downstream tasks by 0.62 points in the hybrid setting.

### 2. OS-Omni: Benchmarking Generalist Computer-Using Agents on Long-Horizon Workflows Across Platforms

[Read Paper](https://huggingface.co/papers/2611.34232)

A live cross-platform benchmark evaluates computer-using agents on long-horizon workflows across Windows, Ubuntu, macOS, iOS, Android, and Web, showing that current agents remain far from platform-robust computer use and trail humans on iOS and macOS.

### 3. Solve Vision with Code: Coding Agents Are Stronger Visual Reasoners Than Video Models

[Read Paper](https://huggingface.co/papers/2611.34813)

Video generation models are increasingly evaluated as visual reasoners: given a first frame and a natural-language prompt, can the model produce the video in which the specified task is carried out? The paradigm is considered advantageous because the reasoning is native to the visual medium, unfolding directly in the frames rather than through language or code. Whether this native route is in fact the more reliable path to visual reasoning has not been tested against a symbolic alternative under the same conditions. We therefore pose the same tasks, with identical inputs, output specification and evaluator, to both media: rather than denoising pixels, a coding agent writes a program that renders the answer video, which an unmodified rule-based evaluator scores. On VBVR-Pro-Bench (Xu et al., 2026), a 500-instance suite spanning five cognitive categories with symbolic ground truth, the agent receives the same first frame and prompt as the video model, with no access to ground truth, metadata or generation models, and one attempt per instance. Three closed-model agents (Codex with gpt-6-astra, Claude Code with Fable 5.1 and Gemini CLI with Gemini 3.1 Pro) score 0.923, 0.894 and 0.893, each more than 0.22 above the strongest video model. Among 23 runnable open-weight models, the best, GLM-5, reaches 0.572, above all video models except the RL-trained one. Together, we show that for tasks whose solutions can be computed from a first frame and a prompt, program synthesis is currently a substantially more reliable route to visual reasoning than video generation. We further distinguish depictive programs, which navigate the image to recover a transferable scheme, from propositional programs, which operate on hard-coded values, revealing how visual reasoning unfolds in agent-written programs and what it implies for its nature of reasoning in general.

### 4. What a Forecasting Leaderboard Can Establish: Resolution Reporting for Zero-Shot Time-Series Models

[Read Paper](https://huggingface.co/papers/2611.31619)

A tool called Vernier measures the smallest score difference a time-series forecasting leaderboard can distinguish, showing that most boards cannot separate top entries and that ranking order can change with pool composition.

### 5. ProxyGenBench: Benchmarking Proxy-Guided Video Generation

[Read Paper](https://huggingface.co/papers/2611.34457)

A benchmark assesses how well video generators follow animated 3D blockouts while preserving structure, filling missing content, correcting proxy artifacts, and binding multimodal conditions to the correct entities and events.

### 6. StateWeaver: Multi-Agent Coordination of Story-State Transitions for Long-Horizon Video Generation

[Read Paper](https://huggingface.co/papers/2611.34290)

StateWeaver improves long-video generation by preserving cross-shot consistency and updating visual references through active state tracking and hierarchical collaboration.

### 7. ShowNow: Fine-Grained Speculative Streaming for Responsive World Models

[Read Paper](https://huggingface.co/papers/2611.34693)

A fine-grained speculative streaming framework prepares action-conditioned video futures during playback to reduce input-to-display latency in interactive world models while sustaining real-time generation.

### 8. DRBench Pro: A Knowledge-Centric Benchmark for Professional Deep Research Agents

[Read Paper](https://huggingface.co/papers/2611.34852)

Deep research agents gather information from multiple sources and synthesize it into reports on complex, open-ended questions. Existing benchmarks assess report quality along several dimensions, but lack a dedicated knowledge-centric evaluation of which research directions a report covers and how deeply it explores them. We introduce DRBench Pro, a benchmark that makes structured knowledge coverage the primary evaluation target. Our automated pipeline builds local knowledge trees from Wikipedia and uses sampled subgraphs to generate research queries and evidence-grounded rubrics. Each rubric corresponds to a relation in the tree, preserving the structure of the knowledge being evaluated. Agents receive only the research query. This structure allows us to measure breadth across major research directions and depth along related knowledge chains, with their geometric mean giving overall coverage. The benchmark includes 144 cases across 24 domains and 44,102 rubrics. Evaluating 16 deep research agents reveals different strengths in breadth and depth, with the best-performing agent reaching 57.04% overall coverage. Rubric judgments agree closely with human annotations, and agent rankings remain stable across the tested metric configurations.

### 9. Boundary-Aware Policy Learning for Agent-Orchestrated Robot Manipulation

[Read Paper](https://huggingface.co/papers/2611.32238)

A method that explicitly shapes and expands the handoff boundary of visuomotor policies improves coordination between multimodal agents and robot policies, yielding higher success and lower cost.

### 10. Revisiting LLMs as Context Engineers

[Read Paper](https://huggingface.co/papers/2611.30494)

As large language models (LLMs) are used in more and more new tasks, context engineering becomes a fundamental method to efficiently steer LLMs by constructing instruction prompts or metadata without intensive training. A growing number of LLM context engineering (LCE) approaches automate the task by casting the LLM itself as the engineer that draws generalizable instructions or experiences from offline trials as context. Although these methods resemble how humans learn from experiences, it remains unclear when and how LLMs can engineer effective context from data. To answer the question, we first establish a benchmark composed of datasets with different reasoning patterns. We compared different LCE methods with different LLM families in 8B size, showing that there is no single winner for all tasks, even compared with simple baselines like chain-of-thought. Even when scaling model capabilities via more parameters or generation iterations, LCEs do not consistently boost existing performance. Across all datasets and models, the performance gains of LCEs are negatively correlated (r=-0.5, ple 0.05) with the diversity of the failure modes of reasoning trajectories. Investigating how LCEs work, we find that (1) the improvement in instruction adherence can explain some success of LCEs; (2) one major bottleneck is whether LLMs can extract effective experiences out of natively diverse failures. Through this study, we uncover the limited capabilities of LCEs in abstracting principled contexts from diverse failure modes and understand how LCEs work, providing guidelines for future work.
