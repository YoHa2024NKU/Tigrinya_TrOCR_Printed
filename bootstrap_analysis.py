import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import json
import numpy as np
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
CACHE_FILE = "bootstrap_predictions_cache.json" # To avoid re-running GPU inference
OUTPUT_IMAGE = "bootstrap_distributions.png"

# Bootstrap Settings 
N_BOOTSTRAP_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_or_run_inference():
    """
    Runs inference on the full test set OR loads from cache if available.
    """
    if os.path.exists(CACHE_FILE):
        print(f"📂 Loading cached predictions from {CACHE_FILE}...")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['preds'], data['refs']

    print(f"🚀 Starting Inference on {device} (This runs once)...")
    
    # Load Model
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    # Load Data
    df = pd.read_csv(TEST_FILE, sep='\t')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))
    
    preds = []
    refs = []

    print(f"   Processing {len(df)} images...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                #  OPTIMIZED SETTINGS
                generated_ids = model.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=5,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
            
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            preds.append(pred_text)
            refs.append(str(row['text']))
            
        except:
            continue

    # Save to cache
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump({'preds': preds, 'refs': refs}, f)
        
    return preds, refs

def perform_bootstrap(preds, refs):
    print(f"\n🔄 Running {N_BOOTSTRAP_SAMPLES} Bootstrap Iterations...")
    
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    cer_scores = []
    wer_scores = []
    acc_scores = []
    
    n_samples = len(preds)
    
    # Convert to numpy for faster indexing
    np_preds = np.array(preds)
    np_refs = np.array(refs)

    for _ in tqdm(range(N_BOOTSTRAP_SAMPLES)):
        # 1. Resampling with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        sample_preds = np_preds[indices]
        sample_refs = np_refs[indices]
        
        # 2. Metric Computation
        # Note: We calculate raw numbers here for speed
        cer = cer_metric.compute(predictions=sample_preds, references=sample_refs)
        wer = wer_metric.compute(predictions=sample_preds, references=sample_refs)
        
        # Exact Match Accuracy
        matches = np.char.strip(sample_preds) == np.char.strip(sample_refs)
        acc = np.mean(matches)
        
        cer_scores.append(cer * 100)
        wer_scores.append(wer * 100)
        acc_scores.append(acc * 100)

    return cer_scores, wer_scores, acc_scores

def plot_distributions(cer_scores, wer_scores, acc_scores):
    print("\n📊 Generating Visualization...")
    sns.set_style("whitegrid")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        (cer_scores, "CER (%)", "Character Error Rate", "#e74c3c"),
        (wer_scores, "WER (%)", "Word Error Rate", "#f39c12"),
        (acc_scores, "Accuracy (%)", "Exact Match Accuracy", "#2ecc71")
    ]
    
    for i, (scores, xlabel, title, color) in enumerate(metrics):
        ax = axes[i]
        
        # Calculate Stats
        mean_val = np.mean(scores)
        lower_ci = np.percentile(scores, 2.5) # 95% CI Lower
        upper_ci = np.percentile(scores, 97.5) # 95% CI Upper
        
        # Plot Histogram
        sns.histplot(scores, kde=True, ax=ax, color=color, edgecolor='black', alpha=0.6)
        
        # Add Lines
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
        ax.axvline(lower_ci, color='blue', linestyle=':', linewidth=2)
        ax.axvline(upper_ci, color='blue', linestyle=':', linewidth=2, label='95% CI')
        
        # Add Confidence Interval Text
        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        
        # Dynamic Text placement
        text_y = ax.get_ylim()[1] * 0.9
        ax.text(mean_val, text_y, f" 95% CI:\n [{lower_ci:.2f}, {upper_ci:.2f}]", 
                ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Saved chart to {OUTPUT_IMAGE}")
    plt.show()

    # Print Final Report for Thesis Text
    print("\n" + "="*50)
    print("🎓 THESIS STATISTICAL REPORT (Copy this)")
    print("="*50)
    print(f"Metric    | Mean   | 95% Confidence Interval")
    print("-" * 50)
    print(f"CER       | {np.mean(cer_scores):.2f}%  | [{np.percentile(cer_scores, 2.5):.2f}%, {np.percentile(cer_scores, 97.5):.2f}%]")
    print(f"WER       | {np.mean(wer_scores):.2f}%  | [{np.percentile(wer_scores, 2.5):.2f}%, {np.percentile(wer_scores, 97.5):.2f}%]")
    print(f"Accuracy  | {np.mean(acc_scores):.2f}% | [{np.percentile(acc_scores, 2.5):.2f}%, {np.percentile(acc_scores, 97.5):.2f}%]")
    print("="*50)

if __name__ == "__main__":
    preds, refs = load_or_run_inference()
    cer, wer, acc = perform_bootstrap(preds, refs)
    plot_distributions(cer, wer, acc)