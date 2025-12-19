<div align="center">

# TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering

<a href="https://arxiv.org/abs/2512.16270"><img src="https://img.shields.io/badge/Paper-arXiv%3A2512.16270-b31b1b?logo=arxiv&logoColor=red"></a>
<a href="https://math-finding.github.io/TextEditBench"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-8A2BE2"></a>
<a href="https://huggingface.co/datasets/MATH-finding/TextEditBench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue"></a>

</div>

## 📢 News
* **[2025-12-19]** 🚀 We have released the evaluation code and the **TextEditBench** dataset!

---

## 📖 Introduction
**TextEditBench** is a comprehensive benchmark for evaluating Reasoning-aware Text Editing beyond mere rendering. TextEditBench explicitly focuses on text-centric regions across 14 topics and 6 task types, emphasizing **reasoning-intensive scenarios** that require models to understand physical plausibility, linguistic meaning, and cross-modal dependencies.  

To comprehensively assess model performance across diverse editing contexts, we establish a Dual-Track Evaluation Framework encompassing **Pixel-Level Objective Metrics** and **MLLM-based Semantic Metrics**. Besides, we propose a novel evaluation dimension, **Semantic Expectation (SE)**, to measure the model's ability to maintain semantic consistency, contextual coherence, and cross-modal alignment.Our approach offers a scalable and reproducible alternative to human evaluation, while maintaining a high degree of alignment with human judgment regarding complex reasoning chains.    


<img src="assets/overview.jpg" width="100%" alt="TextEditBench Overview">

###  ✨ Key Features
* **🧠 Reasoning-Centric:** Introduces **Semantic Expectation (SE)** metric .
* **🌍 Diverse Scenarios:** Covers **14 topics** and **6 task types**.
* **📏 Comprehensive Evaluation:** 
    * **Track 1 (Pixel-level):** SSIM, PSNR, LPIPS, MSE.
    * **Track 2 (Semantic-level):** Powered by **GPT-4o**, evaluating Instruction Following, Text Accuracy, Visual Consistency, Layout Preservation, and Semantic Expectation .

---

## 📊 Dataset Overview  

TextEditBench comprises **1,196 high-quality instances**, curated through a rigorous **Human-AI-Human** verification pipeline. The dataset balances diversity and annotation fidelity by combining **Manual Production (58%)** with **Web-sourced instances (42%)**.

<div align="center">
  <img src="assets/data_distribution.jpg" width="90%" alt="Data Distribution"> 
</div>

### 🧩 Dataset Composition  
* **14 Diverse Topics:** Broad coverage of daily visual contexts, including Professional Documents, Digital Interfaces, Signage, Menus, and Packaging.
* **6 Atom Operations:** Systematic editing tasks designed to test specific capabilities: **Delete, Insert, Change, Relocation, Scaling,** and **Attribute** transfer.  
* **Hierarchical Difficulty:** Each instance is scored (0-20) based on **10 difficulty attributes** and categorized into **Easy, Medium, and Hard** tiers, enabling fine-grained analysis of model robustness.

---

## 🚀 Quick Start

Here we provide a quick start guide to evaluate Models on TextEditBench.

### Setup Environment

```
git clone https://github.com/MATH-finding/TextEditBench.git

conda create -n textedit python=3.10
conda activate textedit
pip install -r requirments
```

Setup API key and API base URL in `.env` for gpt4o.

```
OPENAI_API_KEY=${your_api_proxy_provider_key}
OPENAI_API_URL=${your_ark_api_base_url}
```

### Download Data

Download the TextEditBench data from [Huggingface](https://huggingface.co/datasets/MATH-finding/TextEditBench) and unzip it under the root directory.

```
wget 
unzip data.zip
```

The file structure should be like this:

```
data/                                          
├── data_manual_production/                         
│   └── Art_Creative_Expression/
│   │   ├── 001/
│   │   │   ├── 1.jpg
│   │   │   ├── 1_mask.jpg
│   │   │   ├── Art_Creative_Expression_001.json
│   │   │   ├── text_delete_1.jpg
│   │   │   └── text_delete_1_mask.jpg
│   │   └── ...
│   │
│   └── ...
│
└── data_web-sourced_instances/                         
    ├── Art_Creative_Expression/               
    │   ├── 001/
    │   │   ├── 1.jpg
    │   │   ├── 1_mask.jpg
    │   │   ├── Art_Creative_Expression_001.json
    │   │   └── text_delete_1_mask.jpg
    │   └── ...
    │
    └── ...
```

## 🛠️ Usage

### **Model Output Folder**

```
edited_images/                                          
├── canva/                         
│   └── Art_Creative_Expression/
│   │   ├── 001_edited.jpg
│   │   └── ...
│   └── ...
│
└── real/                         
    ├── Art_Creative_Expression/               
    │   ├── 001_edited.jpg
    │   └── ...
    │
    └── ...
```

### Evaluation

**Track 1 (Pixel-level):** SSIM, PSNR, LPIPS, MSE.

```
python evaluation/masked_mse_psnr_evaluation.py path/to/your_model_output.json

python evaluation/masked_ssim_lpips_evaluation.py path/to/your_model_output.json
```

**Track 2 (Semantic-level):** Powered by **GPT-4o**, evaluating Instruction Following, Text Accuracy, Visual Consistency, Layout Preservation, and Semantic Expectation .

```
python evaluation/gpt_eval.py \
    --input_dir path/to/input_dataset \
    --pred_dir path/to/model_predictions \
    --output_dir results/ \
    --workers 10
```

```
python evaluation/gpt_eval_no_refer.py \
    --input_dir path/to/input_dataset \
    --pred_dir path/to/model_predictions \
    --output_dir results/ \
    --workers 10
```

```
python calculate.py path/to/your_result.json
```



---

## 📝 Citation

If you find our work or dataset useful, please cite us:

```bibtex
@misc{gui2025texteditbenchevaluatingreasoningawaretext,
      title={TextEditBench: Evaluating Reasoning-aware Text Editing Beyond Rendering}, 
      author={Rui Gui and Yang Wan and Haochen Han and Dongxing Mao and Fangming Liu and Min Li and Alex Jinpeng Wang},
      year={2025},
      eprint={2512.16270},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16270}, 
}
```

## 📧 Contact

For any questions, please feel free to open an issue or contact [email@example.com](mailto:email@example.com).

