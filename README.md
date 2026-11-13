# 🧠 MedVision-LLM: Sovereign Medical Vision-Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-00f3ff.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_CUDA_12.4-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Trained_on-RTX_3060_GPU_(FP16_AMP)-76b900.svg?logo=nvidia&logoColor=white)](#)
[![Parameters](https://img.shields.io/badge/Parameters-6.4M_Multimodal_VLM-8b5cf6.svg)](#)
[![Weights](https://img.shields.io/badge/Model_Weights-24.5_MB_(Zero_Dataset_Download)-10b981.svg)](#)

**MedVision-LLM** is a lightweight, sovereign **Medical Vision-Language Model (Med-VLM)** engineered and trained completely **from scratch** on 10,287 Brain MRI neuroimaging scans. It analyzes MRI scans in real-time, predicts pathology probabilities, and generates structured neurological reports (*"what happened, primary diagnosis, severity level, and neurosurgical recommendations"*).

> 🌐 **Try It in the Browser (No Heavy Downloads Needed)**:
> You do **not** need to download gigabytes of raw medical datasets to test this model. The repository comes bundled with compact inference weights (**24.5 MB**) and an interactive Cyberpunk Web Dashboard with preloaded benchmark test scans.

---

## 🚀 Quick Web Test (Instant in Browser)

### 1. Clone & Install Minimal Requirements
```bash
git clone https://github.com/subashis-dev/medvision-llm.git
cd medvision-llm
pip install -r requirements.txt
```

### 2. Launch Interactive Dashboard
```bash
python server.py
```
Open **[http://localhost:8086](http://localhost:8086)** in your browser:
* 🔴 **Glioma Tumor** → Click to test intra-axial brain parenchymal tumor
* 🟠 **Meningioma** → Click to test extra-axial dural-based tumor
* 🔵 **Pituitary Macroadenoma** → Click to test sellar / suprasellar lesion
* 🟢 **Normal Brain MRI** → Click to test healthy brain control
* Or drag & drop your own MRI scan to view the **Live 4-Class Probability Matrix** and generated clinical dossier.

---

## 🧬 Multimodal Architecture

MedVision-LLM combines an inverted-bottleneck **ConvNeXt Visual Feature Extractor** with an **Auxiliary Visual Classifier Head** and a unified **Causal Multimodal Transformer Decoder (SLM)**:

```text
       Medical Scan (Brain MRI Axial / Sagittal Slice)
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │ 1. ConvNeXt Visual Feature Backbone                    │
 │    • 7x7 Depthwise Conv + GroupNorm                    │
 │    • Inverted Bottleneck (4x Expansion with GELU)      │
 │    • 2-Stage Downsampling -> 4x4 Spatial Pooling       │
 └────────────────────────────────────────────────────────┘
            │                               │
            ▼ (Pooled Feats)                ▼ (16 Visual Tokens)
 ┌───────────────────────┐   [ <vis_0> ... <vis_15>, Prompt Tokens... ]
 │ Auxiliary Classifier  │                  │
 │ • Direct Cross-Entropy│                  ▼
 │ • 4 Pathology Classes │   ┌────────────────────────────────────────┐
 └───────────────────────┘   │ 2. Multimodal Transformer Decoder (SLM)│
            │                │    • 6 Layers, 8 Attention Heads       │
            │                │    • Pre-RMSNorm Normalization         │
            │                │    • Scaled Dot-Product Attention      │
            │                │    • Tied Linear LM Projection Head    │
            │                └────────────────────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼ (Autoregressive Clinical Report)
 {
   "diagnosis": "Pituitary Adenoma",
   "severity": "Moderate",
   "finding": "expanded sella turcica with homogeneous suprasellar mass extending toward chiasm",
   "modality": "Brain MRI",
   "clinical_case": "Patient presented with bitemporal hemianopsia and endocrine screening irregularities.",
   "recommendations": "Endocrine hormone panel and neurosurgical consultation for transsphenoidal resection."
 }
```

---

## 📊 Training & Dataset Benchmark

* **Total Scans:** 10,287 real Brain MRI scans (8,582 train / 1,705 validation)
* **Dataset Sources:** Multi-source merged benchmarks (`sartajbhuvaji/Brain-Tumor-Classification-DataSet` + `Brandon-Kee/NeuroScan`)
* **Augmentation:** Dynamic medical online augmentation (rotation $\pm 14^\circ$, scaling, gamma shift $\gamma \in [0.85, 1.20]$, blur)
* **Model Parameters:** 6,429,956 (6.4M)
* **Inference Speed:** ~15ms vision encoding, ~120ms total generation on RTX 3060 Laptop GPU (also runs smoothly on CPU).

---

## 📁 Repository Structure

```text
medvision-llm/
├── data/
│   ├── samples/             # Curated, verified test scans for 1-click evaluation
│   └── vocab.json           # Specialized medical vocabulary with whitespace preservation
├── models/
│   └── medvision_model.pt   # Compact trained weights (24.5 MB)
├── web/
│   ├── css/                 # Glassmorphic cyberpunk styling
│   ├── js/app.js            # Live dashboard logic & probability matrix renderer
│   └── index.html           # Diagnostic UI
├── model.py                 # PyTorch MedVisionVLM architecture
├── tokenizer.py             # Specialized clinical tokenizer
├── inference.py             # Preprocessing, brain skull auto-cropping & inference
├── server.py                # FastAPI backend & REST endpoints (port 8086)
├── test_eval.py             # Fast terminal verification script
└── requirements.txt         # Lightweight dependencies
```

---

## 🔒 Privacy & Safety Notice

* **100% On-Device Execution:** All inference runs locally in system memory / GPU VRAM. Zero telemetry, zero external API calls.
* **Scope:** MedVision-LLM is trained specifically on brain neoplasms (*Glioma*, *Meningioma*, *Pituitary Adenoma*, and *Normal Controls*). Non-tumorous vascular conditions (e.g. acute ischemic stroke / infarction) lack neoplastic features and are classified as negative for intracranial tumor. Designed for research and educational purposes.

---

## 📄 License

MIT License. Open-source for research and clinical AI experimentation.
