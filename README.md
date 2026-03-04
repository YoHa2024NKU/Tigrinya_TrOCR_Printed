<div align="center">

# Tigrinya TrOCR

**Adapting TrOCR for Printed Tigrinya Text Recognition: Word-Aware Loss Weighting for Cross-Script Transfer Learning**

[![Handwritten Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Handwritten%20Variant-yellow)](https://huggingface.co/Yonatanhaile2026/tigrinya-trocr-handwritten)
[![Printed Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Printed%20Variant-orange)](https://huggingface.co/Yonatanhaile2026/tigrinya-trocr-printed)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)

</div>

---

The first Transformer-based benchmark for printed Tigrinya OCR using the Ge'ez script. We fine-tune [TrOCR](https://arxiv.org/abs/2109.10282) with an extended tokenizer (+230 Ge'ez characters) and **Word-Aware Loss Weighting**, a technique that resolves systematic word-boundary failures caused by applying Latin-centric BPE to a non-Latin script. Both the handwritten and printed TrOCR-base variants are fine-tuned under identical conditions and converge to near-identical performance, confirming that the adaptation methodology rather than the pre-training domain is the dominant factor.

The Tigrinya writing system uses the Ge'ez script (fidel), an abugida comprising 33 base consonants, 7 vowel orders (231 core syllographs), 4 labialized consonant groups, and 8 punctuation marks.

## Results

### Both Variants (full test set, n=5,000)

| Model | CER (%) | WER (%) | Accuracy (%) |
|-------|---------|---------|--------------|
| Handwritten variant | 0.20 | 0.77 | 97.44 |
| Printed variant | 0.20 | 0.70 | 97.60 |

### Ablation: Word-Aware Loss Weighting

| Approach | CER (%) | WER (%) | Accuracy (%) |
|----------|---------|---------|--------------|
| Standard training (no weighting) | 20.06 | 79.03 | 0.02 |
| Word-Aware Loss Weighting | 0.20 | 0.77 | 97.44 |

### Zero-shot Baselines (n=500)

| Model | CER (%) | Accuracy (%) |
|-------|---------|--------------|
| Handwritten (zero-shot) | 130.01 | 0.00 |
| Printed (zero-shot) | 99.13 | 0.00 |

## Project Structure
```
TigrinyaTrOCR/
├── config/ # Hyperparameter configurations (YAML)
├── data/ # Dataset (Train/Test/Dev TSV files)
├── outputs/ # Trained model checkpoints
├── src/ # Source Code
│ ├── data.py # Dataset loading & robust error handling
│ ├── model.py # Tokenizer extension & architecture
│ ├── trainer.py # Optimized trainer with Word-Aware Loss Weighting
│ └── utils.py # Logging & seeding
├── app.py # Flask Web Application
├── train.py # Main training entry point
├── prediction.py # Testing & metrics calculation
├── visualize.py # Graph generation
└── requirements.txt # Dependency list
```

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (8 GB+ VRAM recommended)
- CUDA Toolkit 12.x

### Installation

```bash
git clone https://github.com/YoHa2024NKU/tigrinya-trocr-research.git
cd tigrinya-trocr-research
pip install -r requirements.txt
```
Data Preparation
This project uses the GLOCR (GeezLab OCR Dataset), specifically 20,000 samples from the Tigrinya News text-lines portion. The dataset is publicly available through Harvard Dataverse. Place your TSV files in the data/ directory:

Split	Samples	Proportion
Training	10,000	50%
Validation	5,000	25%
Test	5,000	25%
Training
```
python train.py
Training completes in approximately 2 hours and 20 minutes per variant on a single 8 GB GPU.
```
### Evaluation

```bash
python prediction.py
This runs inference on the held-out test set and reports CER, WER, and exact match accuracy.
```
Web Demo
```
python app.py
Launches a Flask web application for interactive Tigrinya OCR.
```
Using the Pre-trained Models
Both fine-tuned variants are available on Hugging Face:

Handwritten variant:
```
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("Yonatanhaile2026/tigrinya-trocr-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("Yonatanhaile2026/tigrinya-trocr-handwritten")

image = Image.open("your_tigrinya_image.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values, num_beams=5, max_length=128)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

**Printed variant:**

```python
processor = TrOCRProcessor.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed")
model = VisionEncoderDecoderModel.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed")
```
## Model Architecture
Both models fine-tune TrOCR-base (~334M parameters), which pairs a BEiT encoder with a RoBERTa-initialized decoder. The tokenizer vocabulary is extended from 50,265 to 50,495 tokens (+230 Ge'ez characters). Word-Aware Loss Weighting applies a 2.0x penalty to word-boundary tokens during training.

Variant	Base Model	Stage 2 Pre-training
Handwritten	microsoft/trocr-base-handwritten	IAM Handwriting Database
Printed	microsoft/trocr-base-printed	Synthetic printed text

## Known Limitations
Numerals and mixed-script text are the primary failure mode (55 of 128 error samples). The model may struggle with Arabic numerals or Latin characters embedded in Tigrinya text.
Fine-tuned for printed text only; handwritten Tigrinya performance has not been evaluated.
Trained on news domain text; performance on other domains (historical manuscripts, legal, medical) may vary.
Evaluation is on synthetic text-line images; real scanned documents are untested.

## Computational Environment
Component	Specification
GPU	NVIDIA GeForce RTX 5060 (Laptop, 8 GB GDDR7)
CPU	Intel Core i9-14900HX
RAM	32 GB
OS	Windows 11 Pro 24H2
PyTorch	2.6.1
Transformers	4.40.0
CUDA	12.8

Citation
```
@article{medhanie2026tigrinytrocr,
author = {Yonatan Haile Medhanie and Yuanhua Ni},
title = {Adapting TrOCR for Printed Tigrinya Text Recognition:
Word-Aware Loss Weighting for Cross-Script Transfer Learning},
journal = {arXiv preprint arXiv:XXXX.XXXXX},
year = {2026},
url = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). You are free to use, modify, and distribute this work for any purpose, provided you give appropriate credit and state any changes made.

## Authors

**Yonatan Haile Medhanie**
College of Software Engineering, Nankai University



---

<div align="center">
<i>If you find this work useful, please star the repo and cite the paper.</i>
</div>