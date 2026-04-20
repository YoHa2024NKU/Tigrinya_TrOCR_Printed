"""
train_crnn_ctc_baseline.py

Train a CRNN-CTC baseline on the synthetic printed Tigrinya dataset for fair comparison with TrOCR models.
- Loads train/dev/test splits from TSV files
- Uses the same image preprocessing as TrOCR pipeline
- Evaluates on CER, WER, and accuracy
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import editdistance

# ----------------------
# 1. CONFIGURATION
# ----------------------
DATA_ROOT = "data"
TRAIN_TSV = os.path.join(DATA_ROOT, "train", "train.tsv")
DEV_TSV = os.path.join(DATA_ROOT, "dev", "dev.tsv")
TEST_TSV = os.path.join(DATA_ROOT, "test", "test.tsv")
IMG_DIRS = {
    "train": os.path.join(DATA_ROOT, "train", "images"),
    "dev": os.path.join(DATA_ROOT, "dev", "images"),
    "test": os.path.join(DATA_ROOT, "test", "images"),
}
BATCH_SIZE = 32
IMG_HEIGHT = 32
IMG_WIDTH = 256
NUM_WORKERS = 0
EPOCHS = 30
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "crnn_ctc_best.pth"

# ----------------------
# 2. DATASET
# ----------------------
class OCRDataset(Dataset):
    def __init__(self, tsv_path, img_dir, vocab, transform=None):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.img_dir = img_dir
        self.transform = transform
        self.vocab = vocab
        self.char2idx = {c: i+1 for i, c in enumerate(vocab)}  # 0 is blank for CTC
        self.idx2char = {i+1: c for i, c in enumerate(vocab)}
        self.idx2char[0] = ""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, str(row['image']))
        text = str(row['text'])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = [self.char2idx[c] for c in text if c in self.char2idx]
        return image, torch.tensor(label, dtype=torch.long), text

# ----------------------
# 3. VOCABULARY
# ----------------------
def build_vocab(tsv_paths):
    chars = set()
    for tsv in tsv_paths:
        df = pd.read_csv(tsv, sep='\t')
        for t in df['text']:
            chars.update(str(t))
    vocab = sorted(list(chars))
    return vocab

# ----------------------
# 4. MODEL: CRNN-CTC
# ----------------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d((2,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)  # [B, W, C, H]
        x = x.view(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.log_softmax(2)
        return x

# ----------------------
# 5. COLLATE FUNCTION
# ----------------------
def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return images, labels, label_lengths, texts

# ----------------------
# 6. TRAINING & EVAL
# ----------------------
def decode(preds, idx2char):
    preds = preds.argmax(2)
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

def cer(ref, hyp):
    return editdistance.eval(ref, hyp) / max(1, len(ref))

def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    return editdistance.eval(ref_words, hyp_words) / max(1, len(ref_words))

def evaluate(model, loader, idx2char):
    model.eval()
    cer_total, wer_total, acc_total, n = 0, 0, 0, 0
    with torch.no_grad():
        for images, _, _, texts in tqdm(loader, desc="Eval"):
            images = images.to(DEVICE)
            logits = model(images)
            pred_texts = decode(logits.cpu(), idx2char)
            for gt, pred in zip(texts, pred_texts):
                cer_total += cer(gt, pred)
                wer_total += wer(gt, pred)
                acc_total += int(gt.strip() == pred.strip())
                n += 1
    return cer_total/n, wer_total/n, acc_total/n

# ----------------------
# 7. MAIN
# ----------------------
def main():
    # Build vocab
    vocab = build_vocab([TRAIN_TSV, DEV_TSV, TEST_TSV])
    print(f"Vocab size: {len(vocab)}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Datasets
    train_set = OCRDataset(TRAIN_TSV, IMG_DIRS['train'], vocab, transform)
    dev_set = OCRDataset(DEV_TSV, IMG_DIRS['dev'], vocab, transform)
    test_set = OCRDataset(TEST_TSV, IMG_DIRS['test'], vocab, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Model
    model = CRNN(num_classes=len(vocab)+1).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Logging for analysis ---
    log = {
        'train_loss': [],
        'dev_cer': [],
        'dev_wer': [],
        'dev_acc': [],
    }
    best_dev_cer = float('inf')
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, labels, label_lengths, _ in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            label_lengths = label_lengths.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)  # [B, W, C]
            log_probs = logits.permute(1, 0, 2)  # [W, B, C]
            input_lengths = torch.full(size=(images.size(0),), fill_value=logits.size(1), dtype=torch.long).to(DEVICE)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())
        avg_loss = float(np.mean(epoch_losses))
        log['train_loss'].append(avg_loss)
        # Validation
        dev_cer, dev_wer, dev_acc = evaluate(model, dev_loader, train_set.idx2char)
        log['dev_cer'].append(dev_cer)
        log['dev_wer'].append(dev_wer)
        log['dev_acc'].append(dev_acc)
        print(f"[Dev] CER: {dev_cer:.4f} | WER: {dev_wer:.4f} | Acc: {dev_acc:.4f}")
        if dev_cer < best_dev_cer:
            best_dev_cer = dev_cer
            torch.save(model.state_dict(), CHECKPOINT)
            print("[INFO] Saved best model.")
        # Save log after each epoch
        with open("crnn_ctc_training_log.json", "w", encoding="utf-8") as f:
            import json
            json.dump(log, f, indent=2)
    # Test
    model.load_state_dict(torch.load(CHECKPOINT))
    test_cer, test_wer, test_acc = evaluate(model, test_loader, train_set.idx2char)
    print(f"[Test] CER: {test_cer:.4f} | WER: {test_wer:.4f} | Acc: {test_acc:.4f}")
    # Save test results
    test_results = {
        'test_cer': test_cer,
        'test_wer': test_wer,
        'test_acc': test_acc,
    }
    with open("crnn_ctc_results.json", "w", encoding="utf-8") as f:
        import json
        json.dump(test_results, f, indent=2)

if __name__ == "__main__":
    main()
