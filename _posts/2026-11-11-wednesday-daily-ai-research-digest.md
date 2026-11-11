---
layout: post
title: "Daily AI Research Papers - Wednesday, November 11, 2026"
date: 2026-11-11
---

Keywords: weight-space programming, frontier LLM agent, GPT-6 Astra agent, Qwen3.5-0.8B, reasoning loops, gradients, BF16 weights, forward statistics

---

### 1. Tree-Ising: Repairing Language Models with Bit-Level Ising Edits

[Read Paper](https://huggingface.co/papers/2611.14210)

Tree-Ising repairs reasoning loops in LLM weights by solving sparse Ising/QUBO problems over BF16 parameters, improving math benchmark accuracy with fewer weight edits and no gradient or LLM-token cost.

### 2. World In-Context Learning for Agentic Robot Manipulation

[Read Paper](https://huggingface.co/papers/2611.12885)

Trained action policies write manipulation skill into network weights, tying what is learned to the distribution it is learned from. In-context learning (ICL) avoids this coupling but has not matched trained policies, because a demonstration recorded in another scene carries no feedback from the world the robot currently faces. We observe that what an agent learns in context is semantic, recording task steps and object roles rather than visual appearance, so experience gathered in one world transfers to another. Based on this insight, we propose World In-Context Learning (WICL), a paradigm in which a frozen model starts from a demonstration, rehearses the task in a world such as a simulator, and writes the feedback into its context rather than its weights. We build RoboScript, a modular infrastructure that implements WICL for agentic manipulation. From a single demonstration, RoboScript learns each task through rehearsal in simulation, freezes the learned knowledge as a Task Memory, and deploys the frozen memory on new scenes with data isolation that ensures every deployment input is available on a real robot. On all 50 tasks of RoboTwin 2.0, RoboScript reaches 83.4% mean success across clean and randomized settings with one demonstration per task and no gradient updates, exceeding the strongest trained action policy by over 20 points. The method generalizes across visual distributions, improves through reflection, scales with the foundation model, and transfers to a physical dual-arm robot through cross-embodiment memory adaptation. With a world in the loop, a robot can learn a manipulation task in its context as well as it learns one in its weights. All code is available at https://github.com/ArcheOne-AI/RoboScript

### 3. Few-Step Streaming Video Generation via Condition Shifting

[Read Paper](https://huggingface.co/papers/2611.14033)

APEX extends a streaming video generation method by replacing a separate fake-score network with condition shifting to estimate the generated distribution’s velocity field, improving instruction following and visual aesthetics while reducing memory and iteration time relative to DMD.

### 4. Don't Just Trust Your Memory: Evidence-Linked Memory Reduces Memory-Induced Degradation in LLM Agents

[Read Paper](https://huggingface.co/papers/2611.14320)

WikiTrace links each distilled lesson to its source trajectory and rationale, enabling LLM agents to verify and correct memories before acting, improving performance and reducing memory-induced degradation across agentic benchmarks.

### 5. dNaViT: Discrete Codes are General-Purpose Visual Representations

[Read Paper](https://huggingface.co/papers/2611.12426)

A discrete native-resolution vision transformer with grouped product quantization and a dual-branch objective acts as a versatile visual tokenizer, matching or exceeding task-specific baselines across visual understanding, generation, and editing tasks.

### 6. MMDM: Measuring Massive Multitask Decision Making

[Read Paper](https://huggingface.co/papers/2611.14309)

System-1 Decision Models directly make structured probabilistic decisions over query-specified options given a state in a single forward pass, without autoregressively generating reasoning trajectories or textual responses. Recent evaluations cover diverse tasks but often rely on existing benchmarks, where prior exposure to test material can obscure generalization. Moreover, real-world decision making often involves long contexts, large option spaces, and complex logical dependencies, challenges that existing benchmarks do not adequately capture together. Thus, we introduce MMDM (Massive Multitask Decision Making), a benchmark of 5,448 high-quality questions across 16 task categories, combining newly constructed tasks with source-grounded recompositions and rewrites for single-select, multi-select, and soft-label decision making in fresh contexts. It covers realistic scenarios such as agent state tracking, tool and skill selection, preference judgments, and rubric scoring, with option spaces ranging from 2 to 1,024 and inputs extending to 64,817 tokens. With MMDM, we conduct a comprehensive quantitative evaluation of 28 decision models, showing that many perform well on common knowledge yet struggle with interdependent options, constrained decisions, and multi-select tasks. For example, Perplexity Decider-V1.2 reaches 78.75% Hard Single-select Accuracy on world knowledge, but only 53.95% on option logic and 47.95% on constraint planning, while the best Hard Multi-select Accuracy across all models is 36.81%. These results highlight joint reasoning, constraint-aware selection, and complete multi-select recovery as key priorities for decision model development, with MMDM providing a testbed for measuring this progress.

### 7. DexForesight: Grounding Dexterous Policies with Implicit Spatial Foresight

[Read Paper](https://huggingface.co/papers/2611.11166)

Dexterous manipulation requires rich spatial foresight over both the surrounding scene and the robot hand. Existing world-action models often acquire such predictive knowledge through explicit future prediction. We introduce DexForesight, a framework that learns spatial foresight implicitly by injecting dynamics-aware world representations into the latent space of a dexterous policy. During training, DexForesight learns an action-conditioned spatial world model from offline scene depth, kinematic hand geometry, and demonstration trajectories. The world model predicts 3D hand-point motion and visual feature changes, capturing both articulated hand dynamics and scene changes under demonstrated actions, including the motion of fingers occluded from the cameras. We then freeze the world model and align policy queries with its predictive world queries, transferring the learned spatial dynamics into policy latents and using them to condition action generation. At deployment, only the policy is retained, operating on RGB observations, language, and robot state without depth input or explicit future prediction. On 11 DexJoCo simulation tasks, DexForesight improves the overall success rate of the base policy from 49.4% to 60.5%. On three real-robot dexterous manipulation tasks, DexForesight achieves 58.3% success, compared with 50.0% for the strongest baseline. Project page https://dexforesight.github.io/

### 8. FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance

[Read Paper](https://huggingface.co/papers/2611.07464)

Language agents increasingly work for users on the open Web, and finance is where trusting them is hardest. The data must come from the Web, the strategy lives in the user's head, and correctness depends on time, so an agent can fail silently: a strategy the user never asked for, unverified data, or a backtest that uses the future yields a report as convincing as a correct one. Existing benchmarks close the world somewhere, fixing the task, the data, or the timeline, and thereby remove these failures instead of measuring them. We introduce FINCOSMOS, an open-world evaluation paradigm for agents doing financial work. A simulated user holds a hidden strategy sampled from a compositional grammar and reveals it only when asked. The agent receives no data, so it must acquire data from the Web, build its own backtest under a data cutoff, and deliver verifiable results. Grading needs no reference answer. Programs verify point-in-time correctness and recompute every reported metric, the delivered strategy is checked against the hidden specification, and a run is accepted only when the user is satisfied and none of 15 veto checks fails. Across four coding agents and a model without tools in 150 sessions, no session is accepted: every session that the simulated user signed off fails a program check, and agents raise only 54--64\% of the details they were not told. Human experts attribute 88.33\% of failures to the agents. With FINCOSMOS feedback, an agent with persistent memory learned all ten tasks.

### 9. MACaQ: Fast, Model Agnostic Calibrated Quantization for Low-Precision Weights and Activations

[Read Paper](https://huggingface.co/papers/2611.10649)

MACaQ is an MXFP4 W4A4 post-training quantization method that quantizes each linear layer independently using online activation transforms and Hessian-aware refinement, reducing cost while preserving accuracy across language and diffusion models.

### 10. Embodied First Exam

[Read Paper](https://huggingface.co/papers/2611.12177)

Early successes on agentic robotics point to a potential paradigm shift toward using coding agents to control robots. Existing robotics benchmarks are built around visuomotor policy learning, leaving it open whether coding agents can reliably control diverse robots under physical-world constraints. We therefore introduce the Embodied First Exam, a unified, comprehensive, and agent-native evaluation framework where coding agents can choose their own control approach: direct control, code to control, or train a policy to control. We hope it serves as a first exam for coding agents before they enter the physical world, with 1,156 tasks in 35 suites on 47 robots, including 9 new suites that go beyond pick-and-place to spatial puzzles, dexterous manipulation, multi-robot interaction, dynamic whole-body control, and sim-to-sim engineering. Each task can be run in up to three evaluation settings. The Standard Setting mimics the real world: the agent receives only sensor observations and, by default, cannot reset the environment. Its Realtime variant keeps the environment running while the agent reasons. The Privileged Setting gives the agent access to the full simulator state and unlimited resets, allowing it to search for an oracle solution for training purposes. Success rates in this setting serve as an approximate upper bound. On the 456 core tasks, the strongest of eight agents, GPT-6 Astra in Codex, achieves a success rate of 34% in the Standard Setting, leaving much room for improvement. To see where agents fail, we label the stage at which failed attempts go wrong and compare failures across agents, settings, and stage categories.
