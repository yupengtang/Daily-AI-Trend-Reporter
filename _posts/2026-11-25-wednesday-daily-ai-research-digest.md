---
layout: post
title: "Daily AI Research Papers - Wednesday, November 25, 2026"
date: 2026-11-25
---

Keywords: vision-language models, computer-use agents, Reinforcement Learning with Verifiable Beliefs (RLVB), outcome reward, process reward, reward model, critic, LLM judge

---

### 1. Antom 3-in-1 Transformer: A Payment Foundation Model

[Read Paper](https://huggingface.co/papers/2611.25403)

We report the development of Antom 3-in-1 Transformer, a payment foundation model that integrates sequence, tabular, and graph representations for payment risk detection. The model contains approximately 7B parameters, with approximately 1.7B activated per task route. Antom 3-in-1 Transformer is pretrained to learn reusable behavioral and relational representations, primarily on Ant International's internal payment data. In a cross-business transfer experiment, pretraining improves the target business metric by 34.5% over the same architecture trained from scratch. With only 25% of the training labels, the pretrained model outperforms the full-label scratch baseline in Average Precision (AP) and Recall@0.5% on different merchant domains. In a four-month online experiment on Ant International's billion-scale payment platform, the model reduced the overall loss rate by 37.88% relative to XGB-Online.

### 2. Lazy Agents Answer Against Their Own Beliefs: Reinforcement Learning with Verifiable Beliefs Teaches Them to Act on What They Believe

[Read Paper](https://huggingface.co/papers/2611.32884)

A reinforcement learning method rewards agents for delaying commitment until their inferred posterior over answers is sufficiently narrow, improving evidence-based answering across vision-language benchmarks.

### 3. Rewirable Circuits in Transformers with Loom Attention

[Read Paper](https://huggingface.co/papers/2611.30634)

We propose loom attention, which allows a Transformer to dynamically rewire its circuits at every token by interweaving a head's vertical attention over the outputs of earlier heads (to obtain its query, key and value) with its horizontal attention over tokens. We instantiate vertical attention efficiently with normalized and gated linear attention over a per-token matrix memory, which stores the compressed head outputs and serves as a second residual stream. This yields LoomFormer, a dual-stream architecture that splits control and content between the vector and matrix streams. Across model sizes and training token budgets in language modeling and downstream tasks, LoomFormer matches Transformers trained with over 3x compute and recent residual-stream baselines (HC/mHC, MUDDFormer, AttnRes) trained with ~1.7-2.8x. For example, trained on the same 300B tokens with only 2% extra FLOPs, LoomPythia-2.8B achieves ~1.6x the average downstream gain of mHC-Pythia-2.8B over Pythia-2.8B, and matches (0-shot) or surpasses (5-shot) Pythia-12B. Code is available at https://github.com/Caiyun-AI/LoomFormer

### 4. Don't Sleep, Keep Watch: Proactive Perception for Computer-Use Agents

[Read Paper](https://huggingface.co/papers/2611.28947)

NightWatch adds a lightweight streaming watcher to vision-language computer-use agents to keep perception synchronized with dynamic interfaces, reducing latency and wasted time while maintaining or improving task success.

### 5. ACAP: Asynchronous Coding Agents for Efficient Robot Control

[Read Paper](https://huggingface.co/papers/2611.31205)

Coding agents can construct and revise robot behavior during execution, but using one agent for the whole robot can delay otherwise independent work. We introduce Asynchronous Coding Agents as Policy (ACAP), an asynchronous framework that organizes one robot as a team of persistent agents. Each member reasons and acts for a configurable control group, such as an arm with its gripper or the torso and chassis, enabling asynchronous reasoning and execution. However, different robot components often have physical dependencies that require coordination. To address this challenge, ACAP introduces a hyper-agent that assigns tasks, coordinates dependencies, and dynamically allocates control between the lead and member agents. This allows independent work to progress concurrently while supporting tightly coupled manipulation when necessary. Across 72 real-world rollouts covering 25 task/platform settings, ACAP reduces task-balanced time cost by 28.2% compared with a single-agent Direct Policy, with improvements across Fully Parallel, Parallel-Sequential, and Collaborative tasks. Across all eight Collaborative settings, ACAP also achieves lower mean time cost than Fixed Delegation, where only member agents can control their assigned groups. An additional 90-rollout study shows lower mean 30-minute-capped time cost than Direct on all six tasks. We further demonstrate mobile manipulation using the same member interface for chassis and torso control. We present the framework and a real-world rollout dataset containing observations, generated programs, actions, and feedback to support research on how coding agents reason and act in the physical world.

### 6. Self-Evolving Evaluation Environments for Computer-Use Agents

[Read Paper](https://huggingface.co/papers/2611.30962)

CUA-SEE is a closed-loop evaluation environment that co-evolves tasks and executable verifiers to adaptively measure computer-use agents’ capabilities, boundaries, and progress reliability.

### 7. S-Mind: Internalizing Multi-View Geometry for Spatial Intelligence

[Read Paper](https://huggingface.co/papers/2611.28649)

Spatial intelligence, the ability to reason about 3D space from visual observations, becomes essential as multimodal large language models (MLLMs) move into the physical world. Beyond perceiving each view, it rests on two cornerstones of human cognition: fusing multiple views into one coherent 3D scene and mentally transforming that scene to new viewpoints as reasoning unfolds. Yet our spatial probes reveal that MLLMs largely lack both capabilities, even after spatial fine-tuning. We therefore present S-Mind, which internalizes multi-view geometry within the MLLM through a compact set of spatial tokens. It supervises these tokens to hold the fused 3D scene and re-emits them to re-express this scene from every viewpoint the reasoning adopts. Across four spatial benchmarks, S-Mind improves its Qwen3.5 backbones by up to 22.26 points and surpasses frontier general models such as GPT-6-Astra (+0.45), Kimi-K3.1-Pro (+1.97), and GLM-5.3-Flash (+7.80) in average accuracy. Moreover, the same probes confirm that it acquires both capabilities, reaching fusion and transformation accuracies of 65.11% and 82.17%, against at most 16.02% and 37.28% for the baselines. At inference, S-Mind also runs about 100x faster than a tool-based agent on the same backbone. Instilled through these spatial tokens during training, spatial capability becomes intrinsic to the backbone and persists even when the tokens are withheld, offering a promising path toward spatial intelligence in MLLMs.

### 8. PartAgent: Agentic 3D Part Segmentation via Expert Integration and Closed-Loop Refinement

[Read Paper](https://huggingface.co/papers/2611.32554)

A training-free agentic framework integrates pretrained 3D and 2D experts with closed-loop refinement to achieve state-of-the-art 3D part segmentation across multiple benchmarks.

### 9. LingSante MedGuard Technical Report

[Read Paper](https://huggingface.co/papers/2611.25509)

As medical AI is increasingly used for patient consultation and clinical decision support, identifying harmful, non-compliant, or ethically problematic responses is essential for responsible deployment. Assessing these risks requires checking clinical claims against relevant evidence, verifying calculations, and applying medical requirements under their intended conditions. We introduce LingSante MedGuard, a capability-enhanced guardrail that integrates supervision from a medical guardrail teacher model and six specialized medical capability experts into a unified model. The guardrail teacher supervises risk decisions and explanations across medical safety, compliance, and ethics, while the capability experts support medical rule understanding, numerical reasoning, multimodal hallucination detection, agentic safety verification, risk localization, and multimodal entity understanding. We consolidate these sources of supervision through offline supervised fine-tuning on teacher-generated demonstrations, followed by multi-teacher on-policy distillation using feedback on student-generated responses. We further introduce LingSante MedGuardBench, a benchmark comprising 11 risk-detection subsets spanning medical safety, compliance, and ethics. At 4B, 9B, and 27B scales, LingSante MedGuard achieves mean benchmark F1 scores of 82.3%, 82.4%, and 87.4%, respectively. The 4B model also achieves 73.9% mean accuracy across MedHallu and MedHALT-NOTA under retrieval-enabled evaluation. Additional evaluations of the verification components show improvements in evidence-supported consistency, source traceability, and rule-guided rationale quality across model scales. These results support the integration of specialized medical capabilities into a unified guardrail for medical risk assessment and evidence-based medical safety verification.

### 10. Embodied AI Fails the Mirror Test: How a Simple Mirror Breaks Robot Policies

[Read Paper](https://huggingface.co/papers/2611.33010)

A new benchmark shows that many robot manipulation policies perform worse when mirrors are present, often because they confuse reflections with real objects rather than simply colliding with the mirror.
