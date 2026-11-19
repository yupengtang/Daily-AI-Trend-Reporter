---
layout: post
title: "Daily AI Research Papers - Thursday, November 19, 2026"
date: 2026-11-19
---

Keywords: System One models, parallel prediction, single forward pass, frozen vision-language model, visual-semantic knowledge, holistic image generation, global consistency, shared random state

---

### 1. Jeveration: System One Models Now Generate

[Read Paper](https://huggingface.co/papers/2611.24738)

Jeveration repurposes a frozen vision-language model to generate coherent images and perform image-to-image and dense prediction tasks in a single forward pass with fast 512x512 generation.

### 2. The Resolution Hypothesis: Agent Evolution's Key Is Identifying Improvements

[Read Paper](https://huggingface.co/papers/2611.23068)

Agent evolution is often bottlenecked by noisy evaluations that cannot reliably identify better candidate artifacts or rank methods, so progress requires making true improvements detectable and accumulable.

### 3. ALIVE: Making Streaming Video Models Interactive

[Read Paper](https://huggingface.co/papers/2611.21366)

Streaming video LLMs can continuously observe the world, yet often handle user requests in isolation. We introduce ALIVE, an interactive streaming video model supporting proactive monitoring, streaming task coordination, and feedback-seeking interaction. We construct ALIVE-SFT-288K and ALIVE-RL-3K for two-stage training. Contrastive supervised fine-tuning teaches when and how to respond, while reinforcement learning with Behavior-Aligned Credit Assignment (BACA) combines trajectory-level and local advantages to refine decisions, responses, and silence across multi-turn interactions. We also introduce FSI-Bench to evaluate feedback seeking and subsequent task completion. ALIVE achieves the highest overall scores among the compared models on four public benchmarks and FSI-Bench, demonstrating strong streaming understanding and interactive capabilities.

### 4. PROLA: Principal-Orthogonal Low-rank Adaptation for Predictive Spatiotemporal Weather Downscaling

[Read Paper](https://huggingface.co/papers/2611.11318)

High-resolution weather prediction is important for resolving local atmospheric patterns missed by coarse global forecasts. Large pretrained weather backbones have improved global forecasting, but adapting them to produce future high-resolution fields from recent coarse states remains challenging and costly. We formulate this setting as Predictive Spatiotemporal Weather Downscaling (PSWD), a task that jointly refines spatial and temporal resolution while predicting future high-resolution trajectories. To this end, we propose a framework that decomposes prediction into spatial and temporal stages using a shared pretrained backbone, fixed numerical scaffolds, and lightweight residual heads. For backbone adaptation, we introduce PROLA (PRincipal-Orthogonal Low-rank Adaptation), which allocates a fixed low-rank budget between pretrained weights' principal directions and their orthogonal complement. PROLA further rescales rank-1 optimizer updates using gradient signals, improving adaptation without increasing trainable rank. Experiments in the PSWD setting show that PROLA outperforms representative low-rank adaptation baselines. Further experiments across diverse datasets and tasks support the broader applicability of PROLA, showing that its benefits extend beyond the primary weather downscaling setting.

### 5. Game2Player: Scaling Gameplay Experience for Visual Agents through Environment-Player Co-Development

[Read Paper](https://huggingface.co/papers/2611.24154)

Games are a rich testbed for visual agents, demanding perception, long-horizon planning, and precise control. However, collecting human gameplay is costly, and learning from gameplay videos requires inferring missing actions. Existing datasets also cover a limited range of games, often without explicit tasks or verifiable outcomes. To address these gaps, we introduce Game2Player, a framework that co-develops game environments and players to collect diverse and verifiable gameplay experience at scale. In the Game2Player loop, a Builder agent adapts existing web games, exposes their internal state, and defines tasks with state-based progress measures and success checks. A Coding Player and a VLM Player then attempt the tasks and report problems to the Builder for repair, so that play serves as both environment validation and training data collection. With Game2Player, we build 568 verified web games with 3,436 tasks, collect 15.7K trajectories (307 hours of gameplay), and hold out 50 games as the benchmark PlayerBench. To evaluate the collected experience as training data, we train VLM agents with progress-weighted SFT, assigning greater weight to steps that advance task progress. Game2Player-9B improves progress from 14.3% to 31.2% on PlayerBench and from 30.0% to 53.1% on the external GameWorld benchmark. More broadly, our work suggests a path for visual agents to scale their gameplay experience by building, verifying, and learning from existing game worlds.

### 6. CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning

[Read Paper](https://huggingface.co/papers/2611.24654)

CurvDelta and CurvLoRA constrain LLM updates to a low-curvature preservation space, improving retention while maintaining target adaptation across reasoning, code, and unlearning tasks.

### 7. HINT-WAM: Hindsight-Informed Training for World Action Models

[Read Paper](https://huggingface.co/papers/2611.24789)

World action models (WAMs) have shown promising generalization and data efficiency in robot manipulation by coupling action generation with predictions of future visual dynamics. Translating these capabilities into reliable task execution, however, remains a challenge for existing post-training methods. Offline supervised fine-tuning provides dense supervision on expert trajectories, but its flow-matching action expert is trained only on noised demonstration actions and never on the intermediate states produced by its own sampler; task-specific adaptation can also compromise pretrained capabilities. On-policy reinforcement learning trains on the policy's own behavior, yet sparse task rewards make exploration and credit assignment difficult, while dense reward shaping still provides scalar evaluations rather than direct action targets. We introduce HINT-WAM, the first on-policy self-distillation framework that supervises the action expert on its own sampling trajectories using hindsight, the future video of the demonstration, as privileged information available to the WAM itself. For each demonstration observation, the student generates its action through its own denoising trajectory, and at the states along this trajectory the same model acts as a teacher conditioned additionally on this hindsight, providing a conditional action distribution as a dense training target, while the student sees only deployment-time inputs and needs no environmental rewards or simulator interaction. To transfer this supervision to a flow-matching action expert, HINT-WAM distills the teacher into the student with an energy score, a distribution-level objective that matches the teacher's action distribution from the student's own samples rather than regressing its velocity field. The project web is available in https://tuandattt.github.io/hintwam/.

### 8. LightTAR: Light and Fast Target-Aware Representations for Multimodal Tabular Learning

[Read Paper](https://huggingface.co/papers/2611.17600)

LightTAR adapts dimensionality reduction to the prediction task by adding target-aware features from encoder layers via regularized partial least squares, improving multimodal tabular model performance with minimal runtime overhead.

### 9. WorldRuler: Interactive World Modeling under Configurable Rules

[Read Paper](https://huggingface.co/papers/2611.24869)

A rule-graph-based generative world model preserves user-defined rules and world states across saving and loading while accurately realizing rule-induced visual changes on designated objects and attributes.

### 10. Code World Model 2.0: Towards Infinite Controllable Video Generation

[Read Paper](https://huggingface.co/papers/2611.22925)

Visual proxies decouple explicit world-state simulation from photorealistic video synthesis, offering a controllable interface for both offline clip creation and real-time interactive streaming. Yet a practical renderer must address two challenges. Upstream proxies vary from coarse layouts to fine articulated geometry depending on the interaction, and streaming inference must operate in a few causal steps without accumulating drift across self-generated history. We present Invoker-Render, pairing a bidirectional creation model with a causal four-step streaming model under a shared conditioning interface over structural proxies, an appearance reference, and language. Trained on an aligned approximately 200-hour multi-granularity corpus, Invoker-Render honors fine geometric constraints when supplied while synthesizing plausible local motion under coarse proxies, enabling runtime selection of input detail. To bridge bidirectional creation and real-time streaming without compounding step compression and rollout drift, we introduce Progressive Forcing (ProForcing). ProForcing first trains a causal predictor and distills it into a four-step renderer using histories derived from training videos. Training then shifts to continuous self-generated rollouts. A pipelined serving stack supports efficient streaming inference. Experiments show versatile control across proxy families, granularities, and camera paths, consistent gains over recent streaming baselines, and stable 30-minute racing rollouts that preserve scene and subject appearance.
