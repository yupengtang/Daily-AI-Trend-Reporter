---
layout: post
title: "Daily AI Research Papers - Wednesday, November 18, 2026"
date: 2026-11-18
---

Keywords: large language model (LLM), Ascend NPUs, kernel optimization, kernel benchmarks, kernel generation, agents, input configurations, tensor shapes

---

### 1. TeleChat to Xing4: Advancing Frontier Agentic Models on Domestic Chips and Frameworks

[Read Paper](https://huggingface.co/papers/2611.22115)

We present Xing 4, a series of large language models that extends the TeleChat lineage through coordinated advances in model architecture and training infrastructure on Ascend. This report describes the design principles and systems engineering behind the series, spanning architecture selection, large-scale pretraining, and reasoning and agentic post-training. We study how sparse activation, attention design, model depth and width, and speculative decoding balance model capacity with computation, memory access, and execution overhead. Our pretraining infrastructure builds on MindSpore HyperParallel, combining topology-aware hybrid sharding, efficient distributed optimizer execution, and coordinated management of long-sequence workloads and activation memory. CPU affinity and core isolation further reduce host scheduling interference. Together, these mechanisms support the pretraining of a 285B-parameter model on 8,192 Ascend 910C cards, providing a concrete setting for the architectural and systems studies. For post-training, we introduce TeleGym, a unified agent execution platform that connects execution, evaluation, and trajectory construction to policy optimization. Disaggregated asynchronous execution, bounded oversampling, and partial rollout accommodate heterogeneous task durations while preserving trajectory continuity across policy updates. Together, these advances establish an Ascend-oriented training foundation for the continued development of Xing 4 across model scales and increasingly demanding reasoning and agent workloads.

### 2. AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

[Read Paper](https://huggingface.co/papers/2611.17621)

AscendGYM provides a benchmark for assessing kernel optimization capabilities on Ascend NPUs using real-workload test cases, two evaluation modes, and a Speedup-Token Score that balances speedup against token cost.

### 3. Weight-Space Programming Doesn't Need Transformers

[Read Paper](https://huggingface.co/papers/2611.22055)

When large language models write neural network weights without gradient-based training, they tend to design sparse table or associative-memory models that outperform same-sized Transformers with far fewer parameters and faster GPU inference, though pretrained LLMs remain more accurate.

### 4. Multi-Branch Transformer

[Read Paper](https://huggingface.co/papers/2611.19800)

Branching Transformer blocks into parallel parameterized branches, combined with shared KV caching, improves language model quality and inference efficiency compared with standard sequential Transformers.

### 5. ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models

[Read Paper](https://huggingface.co/papers/2611.21099)

Frontier multimodal models can now control robots zero-shot from images, yet many manipulation tasks depend on contact forces that images cannot show. Bringing force to an API-only frozen model raises two challenges. First, contact evolves much faster than an API model can decide, so the system must close the loop on force beneath the model. Second, a frozen model cannot interpret raw force streams, so force must be returned in a form the model can reason about. Therefore, we present ForceHarness, a harness that lets frozen models use force without demonstrations or finetuning. After calibrating hand force from a wrist sensor or joint torques alone, ForceHarness addresses the first challenge with self-terminating force primitives that regulate force on their own and return at a contact event. To address the second challenge, a physics translation layer turns each force trace into physical quantities with uncertainty as the tool result. To evaluate ForceHarness against expert force controllers and learned policies under identical conditions, we also build HiddenPhys, a seven-task MuJoCo benchmark with hidden physical variables. On a real arm without a wrist force sensor, ForceHarness raises the zero-shot success rate of GPT-6 Astra from 3.3% to 90.0%. In simulation, it adds 17.5 percentage points on the public ForceBench suite and 34.3 percentage points on HiddenPhys, surpassing force-aware policies trained on demonstrations on both benchmarks and matching hand-written expert controllers on HiddenPhys. Ablations further show that smaller models make significantly better decisions when given physical quantities rather than raw force values.

### 6. Breath-Dx: A Time Series Classification Benchmark for Disease Screening with eNose

[Read Paper](https://huggingface.co/papers/2611.15348)

A comprehensive benchmark evaluates time-series classification methods on a large clinical breath dataset under realistic temporal splits, showing that sensor drift lowers performance and recurrent models are most consistent across diseases.

### 7. Self-Improvement Memory Enables Continual Recursive Self-Improvement

[Read Paper](https://huggingface.co/papers/2611.18856)

Continual recursive self-improvement lets harness optimizers reuse and revise prior edits across changing benchmarks or policy models by storing rationales, predicted and observed outcomes, and evaluation contexts in a self-improvement memory, improving final scores over continuation baselines.

### 8. TCP4D: A Unified Tool-Center-Point Tracking and Generation Model for Robot Manipulation

[Read Paper](https://huggingface.co/papers/2611.23143)

Robot manipulation videos provide a rich visual experience, yet recovering robot action and gripper-state annotations from RGB observations still remains challenging. Using the embodiment-agnostic tool-center-point (TCP) pose as the action space, we present TCP4D, a unified model in which a shared 4D scene representation both tracks observed TCP motion and generates future TCP trajectories. Given a monocular video and initial 2D TCP queries, the tracker directly recovers metric TCP poses in SE(3) and gripper states alongside scene geometry and camera parameters. The generator samples local features from the same 4D representation along the recovered TCP tracks, then combines them with robot observations and language instructions to predict future TCP states. Our new cross-embodiment augmentation pipeline converts existing RGB videos into demonstrations for alternative embodiments by retargeting the recovered motion and synthesizing the corresponding observations, without recorded source-robot states. Experiments show that the tracker achieves absolute trajectory errors of 7.60 and 16.07 mm on DROID and RoboLab and adapts to a new embodiment with only ten demonstrations. The generator achieves 69.19% mean success on RoboTwin and outperforms well-established policy models on two real-world tasks, while joint training on source and augmented demonstrations improves target-robot success by 41.5% over augmented-only training.

### 9. FastFrames: Video Generation at Interactive Latency

[Read Paper](https://huggingface.co/papers/2611.23226)

Autoregressive video diffusion models generate frames based on new observations and inputs, making them a promising medium for interactive visual experiences. However, existing models are orders of magnitude too slow for fluid interaction. This paper introduces FastFrames, a family of fast autoregressive video diffusion models that achieves interactive latency without compromising quality. We profile the inference pipeline of state-of-the-art models and find that operations often treated as fixed implementation details, such as VAE decoding and KV cache updates, in fact account for over half of end-to-end inference latency. We challenge the assumption that these components are indispensable and replace them with lightweight alternatives. Our 1.3B variant runs at 36 ms per frame, or 28 responses per second (RPS), on a single B200 GPU while matching the generation quality of a state-of-the-art model that is nearly 8x slower. Our 14B variant runs at 45 ms per frame, or 22 RPS, on an 8x B200 GPU node and yields significantly better visual quality and realism. We show that FastFrames enables real-time interaction in applications such as camera-conditioned and physics-conditioned video generation.

### 10. Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration

[Read Paper](https://huggingface.co/papers/2611.22275)

A decentralized framework enables teams of humanoids to coordinate whole-body loco-manipulation over long horizons using low-level multi-agent post-training and a high-level vision-language planner with speech-based leader-follower synchronization.
