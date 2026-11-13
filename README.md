# 🧠 MedVision-LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-00f3ff.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_CUDA_12.4-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Trained_on-RTX_3060_GPU_(FP16_AMP)-76b900.svg?logo=nvidia&logoColor=white)](#)
[![Parameters](https://img.shields.io/badge/Parameters-6.4M_Multimodal_VLM-8b5cf6.svg)](#)
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Local_(Zero_Cloud_APIs)-10b981.svg)](#)

**MedVision-LLM** is a lightweight, sovereign **Medical Vision-Language Model (Med-VLM)** engineered and trained completely **from scratch** to analyze Brain MRI neuroimaging scans and generate comprehensive clinical diagnostic case reports (*"what happened, pathology diagnosis, severity level, and neurosurgical recommendations"*).

Unlike cloud-dependent vision APIs, MedVision-LLM runs **100% locally on consumer hardware** (NVIDIA RTX 3060 / multi-core CPU) with zero patient data transmission or subscription dependencies.

---

## 🧬 Multimodal Architecture

MedVision-LLM merges an inverted-bottleneck **ConvNeXt Visual Feature Extractor** with a unified **Causal Multimodal Transformer Decoder (SLM)**:

```text
       Medical Scan (Brain MRI T1/T2 Axial/Sagittal Slice)
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. ConvNeXt Visual Feature Backbone                    │
 │    • 7x7 Depthwise Conv + GroupNorm / LayerNorm        │
 │    • Inverted Bottleneck (4x Expansion with GELU)      │
 │    • 2-Stage Downsampling -> 4x4 Spatial Pooling       │
 └────────────────────────────────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 2. Multimodal Linear Projection Bridge                 │
 │    • Projects 256 vision channels into Transformer dim │
 │    • Adds learned spatial positional embeddings        │
 └────────────────────────────────────────────────────────┘
                             │
                             ▼ (16 Visual Patch Tokens)
 [ <vis_0> ... <vis_15>, <bos>, "Scan: Brain MRI | Task: Analyze clinical case", <sep> ]
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 3. Unified Causal Multimodal Transformer (SLM)         │
 │    • 6 Transformer Layers (256-dim, 8 Attention Heads) │
 │    • Pre-RMSNorm Normalization                         │
 │    • Scaled Dot-Product Causal Self-Attention          │
 │    • 1024-dim SwiGLU / GELU Feed-Forward Network       │
 │    • Weight-Tied Output Projection LM Head             │
 └────────────────────────────────────────────────────────┘
                             │
                             ▼ (Autoregressive Generation)
 {
   "modality": "Brain MRI",
   "finding": "Infiltrative heterogeneous intra-axial mass with surrounding vasogenic edema",
   "diagnosis": "Glioma",
   "severity": "High",
   "clinical_case": "Patient presented with focal neurological deficit and new-onset seizures. MRI reveals cerebral parenchymal neoplasm with local mass effect.",
   "recommendations": "Urgent neurosurgical consult for stereotactic biopsy, surgical resection, and oncology staging."
 }
```

---

## 📊 Model Specifications

| Parameter | Specification | Purpose |
| :--- | :--- | :--- |
| **Model Type** | Hybrid ConvNeXt + Causal Transformer VLM | Unified vision-language autoregressive modeling |
| **Total Parameters** | **6,396,544 (~6.4M)** | Verified via PyTorch parameter profiler |
| **Visual Backbone** | Inverted Bottleneck ConvNeXt Stem | Spatial lesion and anatomical feature extraction |
| **Visual Tokens** | 16 Spatial Tokens (`4x4` grid) | Compressed dense visual representation |
| **Transformer Depth** | 6 Layers (8 Heads, 32-dim/head) | Autoregressive clinical reasoning & generation |
| **Embedding Dimension ($d_{model}$)** | 256 | Dense vector embedding width |
| **FFN Dimension** | 1,024 ($4 \times d_{model}$) | High-capacity non-linear medical knowledge mapping |
| **Vocabulary Size** | 256 (Aligned to 64) | Medical clinical lexicon + JSON syntax characters |
| **Model Footprint** | **~25 MB** (Inference weights) | 100% self-contained; zero external weight downloads |

---

## 📈 Training Dynamics & Benchmark Convergence

Trained on real Brain MRI scans across 4 verified clinical pathology classes (**Glioma**, **Meningioma**, **Pituitary Adenoma**, and **Normal Brain Scans**):

* **Compute Hardware:** NVIDIA GeForce RTX 3060 Laptop GPU (6.0 GB VRAM)
* **Precision:** FP16 Automatic Mixed Precision (`torch.amp.autocast`)
* **Optimization:** AdamW ($\text{LR}=4 \times 10^{-4} \to 2 \times 10^{-5}$) with Cosine Annealing schedule
* **Convergence Time:** **~2.8 minutes** total for complete convergence
* **Validation Loss:** **0.0304** (Validation Perplexity: **1.03**)

---

## 🚀 Quick Start (Local Setup)

### 1. Clone & Enter Directory
```bash
git clone https://github.com/subashis-dev/medvision-llm.git
cd medvision-llm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(For GPU acceleration on NVIDIA hardware, install PyTorch with CUDA 12.4 support: `pip install torch --index-url https://download.pytorch.org/whl/cu124`)*

### 3. Launch Web Diagnostic Dashboard
```bash
python server.py
```

Open your browser at:
```text
http://localhost:8086
```

---

## 🔬 Interactive Clinical Capabilities

* **Drop & Drag Brain MRI Scans:** Upload any axial, sagittal, or coronal MRI slice directly in the web UI.
* **1-Click Benchmark Test Scans:** Click any of the preloaded clinical samples (Glioma, Meningioma, Pituitary, Normal) for instant evaluation.
* **Custom Clinical Prompts:** Ask specific questions such as:
  * *"Scan: Brain MRI | Task: Analyze clinical case and diagnose"*
  * *"Scan: Brain MRI | Question: Explain the clinical case and what happened"*
  * *"Scan: Brain MRI | Query: What pathology is visible in this scan?"*
* **1-Click Dossier Exporter:** Download formatted `.txt` and `.json` neurological case reports for clinical review.

---

## 📜 Clinical Disclaimer & License

**Disclaimer:** MedVision-LLM is built for sovereign edge artificial intelligence research, educational exploration, and algorithmic benchmarking. It should not be used as a standalone substitute for professional medical diagnosis or board-certified radiologist evaluation.

Licensed under the [MIT License](LICENSE) — Copyright (c) 2026 Subashis Roy Bhowmik.
