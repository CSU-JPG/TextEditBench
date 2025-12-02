<div align="center">

<h1>TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering</h1>

<p align="center">
<a href="https://arxiv.org/"><img src="https://img.shields.io/badge/ArXiv-2026-b31b1b.svg"></a>
<a href="https://github.com/MATH-finding/TextEditBench"><img src="https://img.shields.io/badge/GitHub-Code-blue.svg"></a>
<a href="https://huggingface.co/"><img src="https://img.shields.io/badge/HuggingFace-Dataset-orange.svg"></a>
<a href=""><img src="https://img.shields.io/badge/Project-Website-green.svg"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <strong>Anonymous Authors</strong>
</p>

<p align="center">
  <em>CVPR 2026 Submission</em>
</p>

<img src="assets/overview_01.jpg" width="100%" alt="TextEditBench Overview">

</div>

---

## 📣 News
*   **[2026.02]** 🚀 **TextEditBench** dataset and evaluation code are released!
*   **[2026.02]** 📄 Paper is submitted to CVPR 2026.

---

## 📖 Abstract

Text rendering has recently emerged as one of the most challenging frontiers in visual generation. However, **text editing within images remains largely unexplored**, as it requires generating legible characters while preserving semantic, geometric, and contextual coherence.

To fill this gap, we introduce **TextEditBench**, a comprehensive evaluation benchmark that explicitly focuses on text-centric regions in images. Unlike previous benchmarks that focus on pixel manipulations, TextEditBench emphasizes **reasoning-intensive editing scenarios**. We propose a novel evaluation dimension, **Semantic Expectation (SE)**, to measure the model's ability to maintain semantic consistency, contextual coherence, and cross-modal alignment.

---

## 🏆 Leaderboard

Here we present the performance of representative models on the **Real-World** subset of TextEditBench. For full results, please refer to our [Paper](link-to-paper).

| Model | **Overall Score** (MLLM) 🏆 | Instruction Following | Text Accuracy | Semantic Expectation (SE) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen-Image-Edit** | **18.70** | **3.50** | 3.83 | 2.47 |
| **Seedream** | 18.54 | 3.64 | **3.96** | 2.62 |
| **NanoBanana** | 18.22 | 3.18 | 3.60 | **2.73** |
| **Step1X-Edit-Think** | 16.17 | 3.05 | 3.42 | 1.98 |
| **FLUX.1-Kontext** | 14.93 | 2.53 | 2.94 | 1.23 |

> **Note:** The "Overall Score" is an aggregate metric based on MLLM-assisted evaluation (GPT-4o), scoring out of 25.

---

## 🖼️ Dataset Overview

TextEditBench comprises **1,196** annotated instances covering **14 topics**, **6 task types**, and **12 fine-grained sub-tasks**. It is designed to test models on both visual fidelity and complex reasoning.

### 1. Diverse Editing Scenarios
Our benchmark covers various atomic operations including translation, replacement, attribute changes, and complex layout adjustments.

<div align="center">
  <img src="assets/data_distribution.jpg" width="80%" alt="Data Distribution">
</div>

### 2. Reasoning-Aware Tasks (Semantic Expectation)
A key contribution of TextEditBench is the focus on reasoning. Models must understand the context (e.g., math calculation, logical deduction) to perform the correct edit.

<div align="center">
  <img src="assets/reasoning_examples.jpg" width="100%" alt="Reasoning Examples">
  <p><em>Examples requiring multi-step reasoning: Math calculation, Date adjustment, and Context association.</em></p>
</div>

---

## 🛠️ Evaluation

We provide a comprehensive evaluation toolkit supporting both **Pixel-Level Objective Metrics** (SSIM, PSNR, MSE, LPIPS) and **MLLM-based Semantic Metrics** (GPT-4o).

### Installation
git clone https://github.com/MATH-finding/TextEditBench.git
cd TextEditBench
pip install -r requirements.txt
