---
layout: post
title: "Daily AI Research Papers - Friday, September 04, 2026"
date: 2026-09-04
---

Keywords: adapter, compile by training, neural function, compact interpreter, Program-as-Weights, semantic accuracy, supervised fine-tuning, workspace reconstruction

---

### 1. Compile by Training: Turning Natural-Language Specifications into Local Neural Functions

[Read Paper](https://huggingface.co/papers/2609.04199)

Compile by training converts natural-language specifications into reusable neural functions by distilling teacher-generated examples into small adapters, enabling efficient deployment without remote model dependencies.

### 2. Terminal-Universe: Turning Agent Trajectories into Scalable Terminal Environments

[Read Paper](https://huggingface.co/papers/2609.04148)

Terminal-Universe reconstructs executable workspaces from agent trajectories to synthesize diverse training tasks and improves post-training performance through supervised fine-tuning.

### 3. LLaDA-Image: Building Strong Image Generators with Fully Open Training Recipes

[Read Paper](https://huggingface.co/papers/2609.03796)

LLaDA-Image unifies a 6B diffusion transformer with a frozen vision-language module, using image-only pre-training and a Muon optimizer to generate photorealistic images with precise editing, and is distilled into a fast 2-4 step variant that achieves state-of-the-art open-source results.

### 4. Knowing When Not to Reuse: Conditional Experience Transfer in Autonomous LLM Post-Training

[Read Paper](https://huggingface.co/papers/2608.26730)

Boundary-Calibrated Intervention Transfer selectively reuses past training evidence by checking contextual applicability and running bounded trials, reducing harmful updates and improving final model quality in autonomous post-training.

### 5. Random Attention: Rethinking KV Cache Eviction for Efficient Reasoning

[Read Paper](https://huggingface.co/papers/2609.03430)

Random eviction of reasoning tokens matches selective KV cache compression because reasoning traces are self-protecting through redundancy, making scoring unnecessary once prompts are preserved.

### 6. LatentPress: Context Compression Beyond Text and Vision

[Read Paper](https://huggingface.co/papers/2609.01507)

LatentPress compresses conversational and document context into continuous memory tokens read directly by a frozen decoder, achieving high compression with faster inference and improved accuracy over text or OCR methods.

### 7. Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM

[Read Paper](https://huggingface.co/papers/2609.04098)

Fully quantizing hybrid LLMs—including recurrent Gated DeltaNet layers—to 4-bit NVFP4 preserves accuracy across long-context and reasoning benchmarks by localizing outliers and exploiting robust delta-rule dynamics.

### 8. Rethinking On-Policy Distillation of Large Language Models II: One Training Example

[Read Paper](https://huggingface.co/papers/2609.04172)

On-policy distillation improves over hundreds of steps from a single query by rapidly covering teacher states, yet student alignment remains slow, indicating the method is algorithm-starved rather than data-starved.

### 9. Puffin-World: Scaling a Unified Multimodal Model with Native 3D World States

[Read Paper](https://huggingface.co/papers/2609.04196)

Puffin-World is a unified multimodal framework that jointly models physics, geometry, and appearance for physically consistent 3D world generation, reconstruction, and closed-loop exploration.

### 10. Scal3R: Learning Efficient Multi-Relative Pose Query for Scalable Online 3D Reconstruction

[Read Paper](https://huggingface.co/papers/2609.04201)

Scal3R improves long-video online 3D reconstruction by querying multi-reference relative poses with lightweight tokens and pose-graph optimization, reducing drift without retraining the backbone.
