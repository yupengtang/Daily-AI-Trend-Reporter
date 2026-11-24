---
layout: post
title: "Daily AI Research Papers - Tuesday, November 24, 2026"
date: 2026-11-24
---

Keywords: AI research, machine learning, deep learning

---

### 1. StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning

[Read Paper](https://huggingface.co/papers/2611.27915)

StereoVLM is a geometry-aware binocular vision-language model trained with hierarchical supervision from calibrated stereo to enable metric 3D spatial reasoning, evaluated by Stereo3D-Bench.

### 2. Taking Apart and Putting Together: Tracking Objects Through Physical Transformations

[Read Paper](https://huggingface.co/papers/2611.27398)

A training-free framework tracks objects through physical assemblies and disassemblies by generating and evaluating split/merge proposals and querying a multimodal large language model for verification and labeling, achieving state-of-the-art accuracy and faster performance on a new benchmark.

### 3. Optimal Linear Regression Without a Variance

[Read Paper](https://huggingface.co/papers/2611.27983)

We study high-probability linear regression in the infinite variance regime, wherecthe regression noise has only a finite conditional α-moment ν for some unknown αin(1,2), and the covariates may only have finite moments up to a given order (e.g., 4th moment). We give a black-box reduction that converts finite-variance regression procedures into weak-moment procedures. The reduction is based on empirical residual pruning and is agnostic to the moment exponent, the noise scale, the covariance, and the design moment. Under a finite fourth-moment condition on the covariates, any finite-variance subroutine attaining the sub-Gaussian rate yields the following when the number of samples n is above a suitably defined threshold: Without knowing α or ν, we have with probability at least 1-δ that  |widehatθ-θ^*|_Σ
  lesssim_α
  ν^{1/α}
  big(d+log(1/δ){n}big)^{α-1α},  where d is the dimension, Σ is the covariate covariance matrix, and |v|_Σ:=(v^topΣv)^{1/2}. We further prove a matching minimax lower bound that holds under the same conditions. The reduction is computationally efficient whenever the finite-variance subroutine is. We also obtain a distribution-free prediction guarantee under a bounded raw α-moment assumption on the response, and give an analogous α-agnostic reduction for mean estimation.

### 4. A Methodologically Rigorous Benchmark of Linear Regression,RNN, and LSTM for S&P 500 Forecasting:Identifying Sources of Performance Inflation in Financial DeepLearning

[Read Paper](https://huggingface.co/papers/2611.27978)

A corrected evaluation protocol shows that common deep-learning stock-prediction practices inflate results, and benchmarks linear, RNN, and LSTM models against a naive persistence baseline.

### 5. MemOnce: A Scalable Memory Architecture for Large Language Models

[Read Paper](https://huggingface.co/papers/2611.27973)

Deploying large language models (LLMs) in long-horizon and continual interactions requires maintaining reusable memory states for future inference, posing challenges for existing architectures. Standard Transformers accumulate layer-wise, token-level KV caches, incurring linear GPU memory growth and quadratic computational complexity as the context expands. In contrast, linear-time architectures compress history into fixed-size states, but struggle to retain fine-grained details. We introduce MemOnce, a foundation architecture that learns scalable memory states for language models. MemOnce is based on two core designs: (a) Memory encoder-decoder: it caches only a small subset of final-layer hidden states as compact memory representations and dynamically decodes them back into KV caches for attention layers; (b) Memory sparse attention: for each continual input, MemOnce adaptively retrieves relevant memory representations, with the sparse indices computed only once across layers. Coupling these two designs enables efficient on-the-fly memory loading and KV cache decoding, drastically reducing the GPU memory footprint and achieving linear-time complexity. We conduct extensive experiments to show that MemOnce outperforms Transformers and other architectures across varying model sizes, context lengths, and training paradigms (pretraining and post-training adaptation). In particular, adapting Qwen3.5-4B with MemOnce reduces memory state cache by 1313times and prefill latency by 6.9times at 1M context, and improves throughput by 3.5times and overall memory performance by +5.77%. Finally, MemOnce generalizes zero-shot to downstream representation tasks, achieving competitive results against specialized models.

### 6. Non-penetrating World Models

[Read Paper](https://huggingface.co/papers/2611.27963)

A benchmark and lightweight framework leverage hidden states from frozen diffusion transformers to detect and prevent object penetration in interactive video world models, reducing physical violations and generalizing zero-shot.

### 7. Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild

[Read Paper](https://huggingface.co/papers/2611.27673)

Hy-MT3, a multilingual translation model family with up to 30B parameters and MoE variants, improves translation accuracy, completeness, constraint adherence, and language coverage, and is optimized for game localization and image translation while releasing a rubric-based evaluation suite.

### 8. GeoScaffold: From Latent to Realized Geometry in Collaborative Detection

[Read Paper](https://huggingface.co/papers/2611.27667)

A training-only auxiliary foreground-mask objective supplies dense spatial supervision to collaborative BEV 3D detectors, improving localization ranking and detection accuracy without changing inference cost.

### 9. Potential emergence in LLM agents: a statistical physics perspective

[Read Paper](https://huggingface.co/papers/2611.27618)

Agent states act as intermediate variables for quantifying and predicting how post-training organizes LLM agent reasoning, with a statistical-physics framework linking global potential, reward, and entropy.

### 10. PromptTTSD: Controllable Text to Spoken Dialogue Synthesis with Open-Ended Text Descriptions

[Read Paper](https://huggingface.co/papers/2611.27601)

A scalable audiobook-based pipeline and a prompt-aware fine-tuned spoken dialogue synthesis model enable zero-shot voice cloning and natural-language style control while improving contextual appropriateness in multi-turn conversations.
