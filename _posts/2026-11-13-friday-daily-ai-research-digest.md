---
layout: post
title: "Daily AI Research Papers - Friday, November 13, 2026"
date: 2026-11-13
---

Keywords: System One models, token generation, SO-Arena, capability spectrum, decision reliability, evidence-grounded evaluation, manual verification, reference labels

---

### 1. SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models

[Read Paper](https://huggingface.co/papers/2611.15777)

SO-Arena evaluates System One models across domains and capability families, revealing that scaling and reasoning gains vary and that high overall accuracy does not ensure reliable decision-making.

### 2. BirdDreams: Joint Perspective and BEV Video Generation for a Plausible, Reactive Driving World Model

[Read Paper](https://huggingface.co/papers/2611.14815)

Driving world models aim to simulate how a scene unfolds under different ego actions. This requires plausible motion of surrounding vehicles as well as precise ego-trajectory control. Learning vehicle motion from perspective video alone is challenging because motion is entangled with appearance and vehicles can be occluded. Prior work therefore conditions video generation on layouts that specify vehicle motion, providing control over surrounding vehicles but requiring their future trajectories as inputs. We introduce BirdDreams, an autoregressive driving world model that jointly generates perspective video and a schematic bird's-eye-view (BEV) video, making scene layout part of the predicted future. The BEV, rendered from scene annotations for training, depicts roads and vehicles as simple shapes at a fixed metric scale, providing a top-down view of their positions and motion with reduced occlusion between vehicles. Perspective video, by contrast, retains scene details that the BEV omits, such as traffic signals and object appearance. Joint attention between two video diffusion transformers, one per stream, lets each stream inform the other during generation. The predicted BEV also provides a geometric interface for ego control: we translate and rotate its latest frame along an input ego trajectory and use the resulting frames as a soft condition that shows how the scene would appear to move if only the ego vehicle moved. At inference, BirdDreams generates both videos from a single camera image without requiring an HD map or vehicle-state inputs. On PhysicalAI-AV and nuScenes, BirdDreams follows input ego trajectories more faithfully than prior driving world models and generates more plausible ego motion without ego-action conditioning. Human evaluation shows that BirdDreams outperforms a perspective-only baseline in generating realistic surrounding-vehicle behavior.

### 3. PhysLoop: Harnessing LLMs' Physics Awareness for Self-Improving Video Generation Models

[Read Paper](https://huggingface.co/papers/2611.16748)

PhysLoop leverages large language models to guide video generation, selection, and correction, enabling physically consistent self-improvement and controllable data construction for video models.

### 4. G-STAR: Group-Scaled Token Capacity for Deeply Compressed Autoregressive Generation

[Read Paper](https://huggingface.co/papers/2611.13004)

Discrete autoregressive models offer a scalable framework for image generation. However, they face a fundamental trade-off: long token sequences preserve representation quality but slow generation, while aggressive spatial compression reduces representation capacity. Based on product quantization, we show that, under a fixed nominal information budget, group size should scale quadratically with the per-axis spatial compression factor. Building on this principle, we propose GSTAR, a spatial--group co-scaling framework that enables \(32\times\) and even effective \(64\times\) compression while preserving strong reconstruction, understanding, and generation performance. Since larger groups introduce intra-group dependencies that must be modeled efficiently, we analyze their impact on inference latency and generation quality and introduce a semi-autoregressive decoder that combines parallel feature computation with lightweight causal modeling. With these designs, our highly compressed models perform competitively with leading models across five representative text-to-image benchmarks (GenEval, DPG-Bench, PRISM, LongText-Bench, and Qwen-Image-Bench). At \(1024\times1024\) resolution, our model delivers a \(20\times\) speedup over prior discrete autoregressive models, generating each image in 2.5 seconds on a single NVIDIA H200. Together, these results show that highly compressed discrete representations can enable high-quality autoregressive generation at low latency.

### 5. Duo: A Reciprocal World Model for Full-Duplex Interaction

[Read Paper](https://huggingface.co/papers/2611.15131)

A reciprocal world model enables real-time full-duplex audiovisual interaction by jointly modeling intent and realization using autoregressive reasoning, synchronized attention, and dual rollout distillation.

### 6. Actor, Judge, or Teacher? Understanding Decision Models in Long-Horizon Agents

[Read Paper](https://huggingface.co/papers/2611.17167)

A decision model that acts poorly can still teach reinforcement learning policies effectively by judging complete action trajectories, improving success on ALFWorld and WebShop.

### 7. Open Vision Banana: A Fully Open Recipe for Generative Visual Perception

[Read Paper](https://huggingface.co/papers/2611.15844)

Recent work suggests that pretrained image generators encode real-world understanding beyond image synthesis. Vision Banana exploits this by framing vision tasks as conditional image editing. However, its data, weights, code, and implementation details are not public. We present Open Vision Banana (OVB), a fully open and reproducible recipe for converting pretrained generators into unified multi-task perception systems. Our analyses identify the design choices that matter most for generalization, including color encoding, prompt design, parameter adaptation, and task balancing. These choices interact, so individually helpful choices can hurt when combined with others. Starting from 4B and 9B parameter open-weight base models and using 2.1M-3.8M training pairs from public datasets, OVB surpasses Vision Banana's reported Cityscapes zero-shot performance (73.9 vs. 69.9 mIoU) and rivals state-of-the-art affine-invariant depth estimators. Its non-zero-shot variants come within five points of Vision Banana on referring and reasoning segmentation but remain further behind on instance segmentation. All models largely retain the base model's image-editing abilities. With legend ensembling, we introduce a test-time scaling method for generator-based perception. It segments an image under several class-color legends and fuses the results by majority vote, correcting unstable color-class bindings and improving mIoU by up to 8.7 points on Cityscapes and 15.0 points on COCO-Object without retraining. We release the dataset construction and processing pipeline, the training and evaluation code, as well as the final, intermediate, and ablation checkpoints. Project website: https://openvisionbanana.github.io

### 8. On the Design of Hyper-Connections: Simplification, Stability, and Scalability

[Read Paper](https://huggingface.co/papers/2611.05984)

A systematic study of Hyper-Connections architectures identifies the simpler efficient design SimpleHC, which removes residual mixing and uses shared memory access to match mHC performance while adding minimal training overhead and improving stability.

### 9. Ropedia Xperience 2: A Gold Standard for Scaling Physical AI Environments and Behaviors

[Read Paper](https://huggingface.co/papers/2611.17060)

Large foundation models such as GPT-6 Astra and Claude Fable 5.1 have recently been shown to control robots directly from visual observations. They nevertheless remain brittle on long-horizon, contact-rich tasks in real-world environments. Further progress depends on two complementary resources: demonstrations of human behavior that supervise how tasks are performed, and interactive environments in which policies can be rolled out, evaluated, and improved in closed loop. Language models have advanced along well-characterized scaling laws; the corresponding question for physical AI is how capability scales with environments and behaviors, which first requires identifying which data to scale and how to produce and reuse such data continuously. To address this question, we introduce Ropedia Xperience 2, comprising 1,589 released metric-scale 3D scans and a collection of 4,230 egocentric recordings of 74 tasks (131 hours). We further show that each capture can be repurposed for robot learning: scenes are converted into physically interactive simulation environments, human motion is retargeted to whole-body robot trajectories, and additional robot behaviors are obtained through virtual teleoperation and automated long-horizon trajectory generation. We propose this joint capture of environments and behaviors as a gold standard for physical AI data collection, laying a data foundation for establishing scaling laws over environments and behaviors.

### 10. TTB: Test-time Baking for Efficient Rendering of Decoder-only View Synthesis Models

[Read Paper](https://huggingface.co/papers/2611.17331)

By baking global scene information from cross-view attention into a lightweight MLP, Test-Time Baking (TTB) enhances the fidelity of novel-view decoding in Large View Synthesis Models while maintaining the efficiency of Test-Time Training.
