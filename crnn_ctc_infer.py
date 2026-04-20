import torch
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from train_crnn_ctc_baseline import CRNN, build_vocab

# --- CONFIGURATION ---
CHECKPOINT = "crnn_ctc_best.pth"
IMG_HEIGHT = 32
IMG_WIDTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LOAD VOCABULARY (from your train/dev/test sets) ---
DATA_ROOT = "data"
TRAIN_TSV = os.path.join(DATA_ROOT, "train", "train.tsv")
DEV_TSV = os.path.join(DATA_ROOT, "dev", "dev.tsv")
TEST_TSV = os.path.join(DATA_ROOT, "test", "test.tsv")
vocab = build_vocab([TRAIN_TSV, DEV_TSV, TEST_TSV])
idx2char = {i+1: c for i, c in enumerate(vocab)}
idx2char[0] = ""

# --- LOAD MODEL ---
model = CRNN(num_classes=len(vocab)+1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# --- IMAGE PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def decode(logits, idx2char):
    preds = logits.argmax(2)
    pred_texts = []
    for p in preds:
        prev = -1
        s = ""
        for idx in p:
            idx = idx.item()
            if idx != prev and idx != 0:
                s += idx2char.get(idx, "")
            prev = idx
        pred_texts.append(s)
    return pred_texts

def infer(image_path):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(image)
        pred_text = decode(logits.cpu(), idx2char)[0]
    return pred_text

if __name__ == "__main__":
    # Example usage
    img_path = "D:\Tigrinya_OCR_Project\data\\test\images\\00001.jpg"  # <-- change this to your image path
    pred = infer(img_path)
    print(f"Predicted text: {pred}")