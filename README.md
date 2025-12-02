# TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering

<div align="center">

<a href="https://arxiv.org/abs/2511.02778"><img src="https://img.shields.io/badge/Paper-arXiv%3A2511.02778-b31b1b?logo=arxiv&logoColor=red"></a>
<a href="https://your-website-url.github.io"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-8A2BE2"></a>
<a href="https://huggingface.co/datasets/your-username/TextEditBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>

</div>

---

## 📢 News
* **[2025-12-]** 🚀 We have released the evaluation code and the **TextEditBench** dataset!

---

## 📖 Introduction
**TextEditBench** is a comprehensive evaluation benchmark that explicitly focuses on **text-centric regions in images**. Beyond basic pixel manipulations, our benchmark emphasizes **reasoning-intensive editing scenarios** that require models to understand physical plausibility, linguistic meaning, and cross-modal dependencies.

We further propose a novel evaluation dimension, **Semantic Expectation (SE)**, which measures reasoning ability of model to maintain semantic consistency, contextual coherence, and cross-modal alignment during text editing. By focusing evaluation on this long-overlooked yet fundamental capability,TextEditBench establishes a new testing ground for advancing text-guided image editing and reasoning in multimodal generation.

<img src="assets/overview.jpg" width="100%" alt="TextEditBench Overview">

### Key Features
* **🧠 Reasoning-Centric:** Introduces **Semantic Expectation (SE)** metric to measure logical consistency.
* **🌍 Diverse Scenarios:** Covers **14 topics** (e.g., Receipt, Menu, Chart, Poster) and **6 task types**.
* **📏 Comprehensive Evaluation:** 
    * **Track 1 (Pixel-level):** SSIM, PSNR, LPIPS, MSE.
    * **Track 2 (Semantic-level):** Powered by **GPT-4o**, evaluating Instruction Following, Text Accuracy, Layout Preservation, and Reasoning.

---

## 📊 Dataset Overview

TextEditBench comprises **1,196 high-quality instances**, curated from both manual designs (Canva) and web-sourced images. 
It covers **14 topics** (e.g., Receipt, Menu, Chart, Poster) and **6 task types**.
<div align="center">
  <img src="assets/data_distribution.jpg" width="90%" alt="Data Distribution"> 
</div>

---

## 🛠️ Usage

show how to use evaluation

---

## 📝 Citation

If you find our work or dataset useful, please cite us:

```bibtex
@article{texteditbench2026,
  title={TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering},
  author={Anonymous Authors},
  journal={CVPR Submission},
  volume={3050},
  year={2026}
}
```

## 📧 Contact

For any questions, please feel free to open an issue or contact [email@example.com](mailto:email@example.com).

```

