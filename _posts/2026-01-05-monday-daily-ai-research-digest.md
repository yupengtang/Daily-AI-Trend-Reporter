---
layout: post
title: "Daily AI Research Papers - Monday, January 05, 2026"
date: 2026-01-05
---

Keywords: 4D world model, 4D reconstruction, novel-trajectory video generation, monocular videos, pose-free, feed-forward, degradation pattern simulation, diffusion forcing

---

### 1. NeoVerse: Enhancing 4D World Model with in-the-wild Monocular Videos

[Read Paper](https://huggingface.co/papers/2601.00393)

NeoVerse is a scalable 4D world model that enables pose-free reconstruction and novel-trajectory video generation from monocular videos with state-of-the-art performance.

### 2. Youtu-Agent: Scaling Agent Productivity with Automated Generation and Hybrid Policy Optimization

[Read Paper](https://huggingface.co/papers/2512.24615)

Existing Large Language Model (LLM) agent frameworks face two significant challenges: high configuration costs and static capabilities. Building a high-quality agent often requires extensive manual effort in tool integration and prompt engineering, while deployed agents struggle to adapt to dynamic environments without expensive fine-tuning. To address these issues, we propose Youtu-Agent, a modular framework designed for the automated generation and continuous evolution of LLM agents. Youtu-Agent features a structured configuration system that decouples execution environments, toolkits, and context management, enabling flexible reuse and automated synthesis. We introduce two generation paradigms: a Workflow mode for standard tasks and a Meta-Agent mode for complex, non-standard requirements, capable of automatically generating tool code, prompts, and configurations. Furthermore, Youtu-Agent establishes a hybrid policy optimization system: (1) an Agent Practice module that enables agents to accumulate experience and improve performance through in-context optimization without parameter updates; and (2) an Agent RL module that integrates with distributed training frameworks to enable scalable and stable reinforcement learning of any Youtu-Agents in an end-to-end, large-scale manner. Experiments demonstrate that Youtu-Agent achieves state-of-the-art performance on WebWalkerQA (71.47\%) and GAIA (72.8\%) using open-weight models. Our automated generation pipeline achieves over 81\% tool synthesis success rate, while the Practice module improves performance on AIME 2024/2025 by +2.7\% and +5.4\% respectively. Moreover, our Agent RL training achieves 40\% speedup with steady performance improvement on 7B LLMs, enhancing coding/reasoning and searching capabilities respectively up to 35\% and 21\% on Maths and general/multi-hop QA benchmarks.

### 3. Taming Hallucinations: Boosting MLLMs' Video Understanding via Counterfactual Video Generation

[Read Paper](https://huggingface.co/papers/2512.24271)

Multimodal Large Language Models (MLLMs) have made remarkable progress in video understanding. However, they suffer from a critical vulnerability: an over-reliance on language priors, which can lead to visual ungrounded hallucinations, especially when processing counterfactual videos that defy common sense. This limitation, stemming from the intrinsic data imbalance between text and video, is challenging to address due to the substantial cost of collecting and annotating counterfactual data. To address this, we introduce DualityForge, a novel counterfactual data synthesis framework that employs controllable, diffusion-based video editing to transform real-world videos into counterfactual scenarios. By embedding structured contextual information into the video editing and QA generation processes, the framework automatically produces high-quality QA pairs together with original-edited video pairs for contrastive training. Based on this, we build DualityVidQA, a large-scale video dataset designed to reduce MLLM hallucinations. In addition, to fully exploit the contrastive nature of our paired data, we propose Duality-Normalized Advantage Training (DNA-Train), a two-stage SFT-RL training regime where the RL phase applies pair-wise ell_1 advantage normalization, thereby enabling a more stable and efficient policy optimization. Experiments on DualityVidQA-Test demonstrate that our method substantially reduces model hallucinations on counterfactual videos, yielding a relative improvement of 24.0% over the Qwen2.5-VL-7B baseline. Moreover, our approach achieves significant gains across both hallucination and general-purpose benchmarks, indicating strong generalization capability. We will open-source our dataset and code.

### 4. Avatar Forcing: Real-Time Interactive Head Avatar Generation for Natural Conversation

[Read Paper](https://huggingface.co/papers/2601.00664)

Avatar Forcing framework enables real-time interactive head avatar generation with low latency and expressive motion through diffusion forcing and label-free preference optimization.

### 5. Nested Learning: The Illusion of Deep Learning Architectures

[Read Paper](https://huggingface.co/papers/2512.24695)

Despite the recent progresses, particularly in developing Language Models, there are fundamental challenges and unanswered questions about how such models can continually learn/memorize, self-improve, and find effective solutions. In this paper, we present a new learning paradigm, called Nested Learning (NL), that coherently represents a machine learning model with a set of nested, multi-level, and/or parallel optimization problems, each of which with its own context flow. Through the lenses of NL, existing deep learning methods learns from data through compressing their own context flow, and in-context learning naturally emerges in large models. NL suggests a philosophy to design more expressive learning algorithms with more levels, resulting in higher-order in-context learning and potentially unlocking effective continual learning capabilities. We advocate for NL by presenting three core contributions: (1) Expressive Optimizers: We show that known gradient-based optimizers, such as Adam, SGD with Momentum, etc., are in fact associative memory modules that aim to compress the gradients' information (by gradient descent). Building on this insight, we present other more expressive optimizers with deep memory and/or more powerful learning rules; (2) Self-Modifying Learning Module: Taking advantage of NL's insights on learning algorithms, we present a sequence model that learns how to modify itself by learning its own update algorithm; and (3) Continuum Memory System: We present a new formulation for memory system that generalizes the traditional viewpoint of long/short-term memory. Combining our self-modifying sequence model with the continuum memory system, we present a continual learning module, called Hope, showing promising results in language modeling, knowledge incorporation, and few-shot generalization tasks, continual learning, and long-context reasoning tasks.

### 6. SenseNova-MARS: Empowering Multimodal Agentic Reasoning and Search via Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2512.24330)

While Vision-Language Models (VLMs) can solve complex tasks through agentic reasoning, their capabilities remain largely constrained to text-oriented chain-of-thought or isolated tool invocation. They fail to exhibit the human-like proficiency required to seamlessly interleave dynamic tool manipulation with continuous reasoning, particularly in knowledge-intensive and visually complex scenarios that demand coordinated external tools such as search and image cropping. In this work, we introduce SenseNova-MARS, a novel Multimodal Agentic Reasoning and Search framework that empowers VLMs with interleaved visual reasoning and tool-use capabilities via reinforcement learning (RL). Specifically, SenseNova-MARS dynamically integrates the image search, text search, and image crop tools to tackle fine-grained and knowledge-intensive visual understanding challenges. In the RL stage, we propose the Batch-Normalized Group Sequence Policy Optimization (BN-GSPO) algorithm to improve the training stability and advance the model's ability to invoke tools and reason effectively. To comprehensively evaluate the agentic VLMs on complex visual tasks, we introduce the HR-MMSearch benchmark, the first search-oriented benchmark composed of high-resolution images with knowledge-intensive and search-driven questions. Experiments demonstrate that SenseNova-MARS achieves state-of-the-art performance on open-source search and fine-grained image understanding benchmarks. Specifically, on search-oriented benchmarks, SenseNova-MARS-8B scores 67.84 on MMSearch and 41.64 on HR-MMSearch, surpassing proprietary models such as Gemini-3-Flash and GPT-5. SenseNova-MARS represents a promising step toward agentic VLMs by providing effective and robust tool-use capabilities. To facilitate further research in this field, we will release all code, models, and datasets.

### 7. Deep Delta Learning

[Read Paper](https://huggingface.co/papers/2601.00417)

Deep Delta Learning introduces a novel residual connection mechanism that uses a learnable geometric transformation to generalize identity shortcuts, enabling more flexible feature modeling while maintaining stable training properties.

### 8. AdaGaR: Adaptive Gabor Representation for Dynamic Scene Reconstruction

[Read Paper](https://huggingface.co/papers/2601.00796)

Reconstructing dynamic 3D scenes from monocular videos requires simultaneously capturing high-frequency appearance details and temporally continuous motion. Existing methods using single Gaussian primitives are limited by their low-pass filtering nature, while standard Gabor functions introduce energy instability. Moreover, lack of temporal continuity constraints often leads to motion artifacts during interpolation. We propose AdaGaR, a unified framework addressing both frequency adaptivity and temporal continuity in explicit dynamic scene modeling. We introduce Adaptive Gabor Representation, extending Gaussians through learnable frequency weights and adaptive energy compensation to balance detail capture and stability. For temporal continuity, we employ Cubic Hermite Splines with Temporal Curvature Regularization to ensure smooth motion evolution. An Adaptive Initialization mechanism combining depth estimation, point tracking, and foreground masks establishes stable point cloud distributions in early training. Experiments on Tap-Vid DAVIS demonstrate state-of-the-art performance (PSNR 35.49, SSIM 0.9433, LPIPS 0.0723) and strong generalization across frame interpolation, depth consistency, video editing, and stereo view synthesis. Project page: https://jiewenchan.github.io/AdaGaR/

### 9. The Reasoning-Creativity Trade-off: Toward Creativity-Driven Problem Solving

[Read Paper](https://huggingface.co/papers/2601.00747)

Large language model training methods that optimize for correctness can cause reasoning path diversity collapse, but a new variational framework provides principled solutions to maintain both accuracy and creativity.

### 10. Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning

[Read Paper](https://huggingface.co/papers/2512.24146)

Researchers identify and address Preference Mode Collapse in text-to-image diffusion models through a novel framework that maintains generative diversity while improving human preference alignment.
