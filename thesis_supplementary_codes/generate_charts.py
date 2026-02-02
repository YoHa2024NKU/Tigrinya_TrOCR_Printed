import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import numpy as np
import evaluate

# CONFIG
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # Load
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    df = pd.read_csv(TEST_FILE, sep='\t')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))
    
    # We need predictions to make charts
    cer_metric = evaluate.load("cer")
    sample_cers = []
    text_lengths = []
    
    print("running inference for statistics...")
    # Run on subset for chart generation speed (e.g. 500 samples)
    df_subset = df.sample(n=min(1000, len(df)), random_state=42)
    
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
            with torch.no_grad():
                ids = model.generate(pixel_values, max_length=128)
            pred = processor.batch_decode(ids, skip_special_tokens=True)[0]
            
            # Calculate per-sample CER
            if len(row['text']) > 0:
                score = cer_metric.compute(predictions=[pred], references=[row['text']])
                sample_cers.append(score * 100) # Convert to %
                text_lengths.append(len(row['text']))
        except:
            pass

    # --- PLOT 1: CER DISTRIBUTION ---
    plt.figure(figsize=(10, 6))
    sns.histplot(sample_cers, bins=20, kde=True, color="#3498db")
    plt.title("Distribution of Character Error Rates (Test Set)")
    plt.xlabel("CER (%)")
    plt.ylabel("Count")
    plt.axvline(x=5, color='r', linestyle='--', label='5% Threshold')
    plt.legend()
    plt.savefig("cer_distribution.png", dpi=300)
    print("✅ Saved cer_distribution.png")

    # --- PLOT 2: ERROR VS LENGTH ---
    plt.figure(figsize=(10, 6))
    plt.scatter(text_lengths, sample_cers, alpha=0.5, color="#9b59b6")
    plt.title("Error Rate vs. Sentence Length")
    plt.xlabel("Character Count")
    plt.ylabel("CER (%)")
    plt.savefig("error_vs_length.png", dpi=300)
    print("✅ Saved error_vs_length.png")

if __name__ == "__main__":
    main()