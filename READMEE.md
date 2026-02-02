
# TigrinyaTrOCR: End-to-End OCR for Ethiopic Script 🇪🇷

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B_(Nightly)-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/State--of--the--Art-94.42%25_Accuracy-success)](outputs/fast_model)
[![Hardware](https://img.shields.io/badge/Hardware-RTX_5060_(Blackwell)-76b900.svg)](https://www.nvidia.com/)

> **Master's Thesis Project**  
> **Author:** Medhanie Yonatan Haile  
> **Institution:** Nankai University, College of Software  
> **Task:** Optical Character Recognition (OCR) for Low-Resource Languages

---

## 📌 Abstract
**TigrinyaTrOCR** is a fine-tuned Transformer-based OCR model designed for the **Tigrinya language** (Ethiopic script). It utilizes the **Microsoft TrOCR** architecture (Vision Transformer Encoder + RoBERTa Decoder) to achieve state-of-the-art results on printed Tigrinya text.

The project identifies and resolves a critical **tokenization mismatch** between pre-trained English BPE tokenizers and disjoint Ethiopic characters. By introducing a novel **"Word-Aware Loss Weighting"** strategy, this model overcomes systematic "character elision" errors, improving exact match accuracy from near-zero to **94.42%**.

### 🚀 Key Features
*   **Word-Aware Loss:** Custom training objective that weights word boundaries by **2.0x** to fix tokenizer conflicts.
*   **Hardware Optimized:** Optimized for **NVIDIA RTX 50-series (Blackwell)** GPUs using Gradient Accumulation (Effective Batch Size 8) and Mixed Precision (FP16).
*   **Interactive Demo:** Includes a Flask-based Web Interface for batch processing and real-time validation.
*   **Extended Vocabulary:** Support for 230+ Ethiopic characters including numerals and punctuation.

---

## 📊 Benchmark Results

Evaluation was conducted on the full **Tigrinya Test Set (N=5,000)**.

| Training Method | CER (%) | WER (%) | Accuracy (%) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Vanilla Baseline** (Zero-Shot) | 118.17 | 112.06 | 0.00 | Failed |
| **Standard Fine-Tuning** | 20.17 | 78.95 | 0.02 | Failed (Eating Letters) |
| **Word-Aware Loss (Ours)** | **0.41** | **1.66** | **94.42** | **State-of-the-Art** |

> *Note: Standard fine-tuning achieved decent character recognition (20% CER) but failed completely on exact sentence matching due to skipping the first letter of every word. The proposed Word-Aware Loss rectified this structural flaw.*

![Training Loss Curve](training_loss_curve.png)
*(Figure: Training convergence over 12,500 steps)*

---

## 🛠️ Installation

**Prerequisites:** Python 3.10+ and an NVIDIA GPU (CUDA 12.x required).

### 1. Clone Repository
```bash
git clone https://github.com/medhanie-yh/TigrinyaTrOCR.git
cd TigrinyaTrOCR
```

### 2. Install PyTorch (Critical)
**Note:** For **RTX 5060 / 50-series** GPUs, you must use PyTorch Nightly (CUDA 12.8) as Stable versions are incompatible with the Blackwell architecture.

```bash
# Windows / Linux
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚡ Usage

### 1. Training
Run the optimized training pipeline. This uses **Batch Size 2** with **Gradient Accumulation 4** to fit on 8GB VRAM while mathematically simulating Batch Size 8.

```bash
python train.py
```
*   **Output:** Model weights saved to `outputs/fast_model/`.
*   **Time:** Approx. 2.5 hours on RTX 5060.

### 2. Evaluation
Generate CER, WER, and Accuracy metrics on the test set.

```bash
python evaluate.py
```

### 3. Visualization
Generate training loss curves and error distribution charts for the thesis.

```bash
python visualize.py
```

### 4. Web Interface (Demo)
Launch the local web app to test images. Features **Batch Upload** and **Green/Red Validation**.

```bash
python app.py
```
*   Open your browser at: `http://localhost:5000`

---

## 📂 Project Structure

```text
TigrinyaTrOCR/
├── config/             # Hyperparameter configurations (YAML)
├── data/               # Dataset (Train/Test/Dev TSV files)
├── outputs/            # Trained model checkpoints
├── src/                # Source Code
│   ├── data.py         # Dataset loading & Robust error handling
│   ├── model.py        # Tokenizer extension & Architecture
│   ├── trainer.py      # Optimized Trainer Logic
│   └── utils.py        # Logging & Seeding
├── app.py              # Flask Web Application
├── train.py            # Main Training Entry Point
├── evaluate.py         # Testing & Metrics Calculation
├── visualize.py        # Graph Generation
└── requirements.txt    # Dependency List
```

---

## 📜 Citation

If you use this code or methodology, please cite the thesis:

```bibtex
@mastersthesis{Haile2026TigrinyaTrOCR,
  author  = {Medhanie Yonatan Haile},
  title   = {End-to-End Optical Character Recognition for Low-Resource Ethiopic Script using Transformers},
  school  = {Nankai University},
  year    = {2026},
  address = {Tianjin, China}
}
```

## 🙏 Acknowledgements
*   **Ministry of Commerce (MOFCOM), PRC:** For scholarship support.
*   **Nankai University:** For academic supervision and resources.
*   **Hugging Face:** For the Transformers library.
```