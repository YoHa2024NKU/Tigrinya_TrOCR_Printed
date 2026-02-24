<div align="center">

# Tigrinya TrOCR

**Adapting TrOCR for Tigrinya: Transfer Learning Strategies for Low-Resource Optical Character Recognition of Ge'ez Script**

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Hugging%20Face-yellow)](https://huggingface.co/Yonatanhaile2026/tigrinya-trocr-printed-model)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)

</div>

---

A fine-tuned [TrOCR](https://arxiv.org/abs/2109.10282) model for **printed Tigrinya line-level text recognition**, achieving **0.20% CER**, **0.77% WER**, and **97.44% exact match accuracy** on a held-out test set of 5,000 samples.

The Tigrinya writing system uses the Ge'ez script (fidel) is an abugida comprising 33 base consonants, 7 vowel orders (231 core syllographs), 4 labialized consonant groups, and 8 punctuation marks.

## Results

| Metric | Value |
|--------|-------|
| Character Error Rate (CER) | 0.20% |
| Word Error Rate (WER) | 0.77% |
| Exact Match Accuracy | 97.44% |
| Perfect Transcriptions | 4,872 / 5,000 |

## Project Structure

```
TigrinyaTrOCR/
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

### Data Preparation

This project uses the **GLOCR (GeezLab OCR Dataset)**, specifically 20,000 samples from the Tigrinya News text-lines portion. Place your TSV files in the `data/` directory:

| Split | Samples | Proportion |
|-------|---------|------------|
| Training | 10,000 | 50% |
| Validation | 5,000 | 25% |
| Test | 5,000 | 25% |

### Training

```bash
python train.py
```

### Evaluation

```bash
python predict.py
```

This runs inference on the held-out test set and reports CER, WER, and exact match accuracy.

### Web Demo

```bash
python app.py
```

Launches a Flask web application for interactive Tigrinya OCR.

### Using the Pre-trained Model

The fine-tuned model is available on Hugging Face:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

processor = TrOCRProcessor.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed-model")
model = VisionEncoderDecoderModel.from_pretrained("Yonatanhaile2026/tigrinya-trocr-printed-model")

image = Image.open("your_tigrinya_image.png").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(text)
```

## Model Architecture

This project fine-tunes [`microsoft/trocr-base-handwritten`](https://huggingface.co/microsoft/trocr-base-handwritten) on printed Tigrinya text, extending the tokenizer to cover the full Ge'ez character set.

## Known Limitations

- **Numerals and mixed-script text** are the primary failure mode (55 of 128 error samples). The model may struggle with Arabic numerals or Latin characters embedded in Tigrinya text.
- Fine-tuned for **printed text only**; handwritten Tigrinya performance has not been evaluated.
- Trained on news domain text; performance on other domains (historical manuscripts, social media) may vary.

## Computational Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 5060 (Laptop, 8 GB GDDR7) |
| CPU | Intel Core i9-14900HX |
| RAM | 32 GB |
| OS | Windows 11 Pro 24H2 |
| PyTorch | 2.6.1 |
| Transformers | 4.40.0 |
| CUDA | 12.8 |

## Citation

<!-- Will be Updated with published thesis/paper details -->

```bibtex
@misc{medhanie2026tigrinya-trocr,
  author    = {Yonatan Haile Medhanie},
  title     = {Adapting TrOCR for Tigrinya: Transfer Learning Strategies
               for Low-Resource Optical Character Recognition of Ge'ez Script},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/YoHa2024NKU/tigrinya-trocr-research},
  note      = {Master's thesis, Nankai University. [Full citation to be added upon publication.]}
}
```

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). You are free to use, modify, and distribute this work for any purpose, provided you give appropriate credit and state any changes made.

## Author

**Yonatan Haile Medhanie** \
Nankai University

---

<div align="center">
<i>If you find this work useful, please star the repo and cite the paper.</i>
</div>