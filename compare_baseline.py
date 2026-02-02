import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import pandas as pd
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np

# --- CONFIG ---
MY_MODEL_PATH = "outputs/fast_model"
VANILLA_MODEL = "microsoft/trocr-base-handwritten"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 500  # Subset size for quick comparison

def evaluate_model(model_name_or_path, df, desc):
    print(f"🔄 Loading {desc} ({model_name_or_path})...")
    try:
        processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path).to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading {model_name_or_path}: {e}")
        # Return dummy high errors if model fails to load
        return 1.0, 1.0, 0.0

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    
    preds, refs = [], []
    
    print(f"   Running Inference on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)
            
            with torch.no_grad():
                # Greedy search is fine for baseline comparison speed
                generated_ids = model.generate(pixel_values, max_length=128)
            
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            preds.append(pred)
            refs.append(row['text'])
        except:
            continue

    # Compute Metrics
    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    acc = sum([1 for p, r in zip(preds, refs) if p.strip() == r.strip()]) / len(refs)
    
    return cer, wer, acc

def main():
    # 1. Load Data
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        return

    df = pd.read_csv(TEST_FILE, sep='\t')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))
    
    # Subsample for speed (comparing 500 images is statistically enough for a baseline chart)
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    # 2. Evaluate Vanilla (The "Before")
    v_cer, v_wer, v_acc = evaluate_model(VANILLA_MODEL, df, "Vanilla TrOCR")
    
    # 3. Evaluate Ours (The "After")
    m_cer, m_wer, m_acc = evaluate_model(MY_MODEL_PATH, df, "TigrinyaTrOCR (Ours)")

    # 4. Print Table
    print("\n" + "="*65)
    print(f"📊 BASELINE COMPARISON (N={SAMPLE_SIZE})")
    print("="*65)
    print(f"{'Metric':<10} | {'Vanilla TrOCR':<20} | {'Ours (Word-Aware)':<20} | {'Improvement':<10}")
    print("-" * 65)
    print(f"{'CER':<10} | {v_cer*100:6.2f}%             | {m_cer*100:6.2f}%             | {((v_cer-m_cer)/v_cer)*100:6.1f}%")
    print(f"{'WER':<10} | {v_wer*100:6.2f}%             | {m_wer*100:6.2f}%             | {((v_wer-m_wer)/v_wer)*100:6.1f}%")
    print(f"{'Accuracy':<10} | {v_acc*100:6.2f}%             | {m_acc*100:6.2f}%             | +{(m_acc-v_acc)*100:6.1f}")
    print("="*65)

    # 5. Generate Chart
    labels = ['CER (Character Error)', 'WER (Word Error)', 'Accuracy (Exact Match)']
    vanilla_scores = [v_cer * 100, v_wer * 100, v_acc * 100]
    our_scores = [m_cer * 100, m_wer * 100, m_acc * 100]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bars
    rects1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla TrOCR (Base)', color='#95a5a6')
    rects2 = ax.bar(x + width/2, our_scores, width, label='TigrinyaTrOCR (Ours)', color='#2ecc71')

    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Baseline Comparison: Impact of Fine-Tuning', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    
    # Add grid lines behind bars
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    # Add numeric labels on top of bars
    ax.bar_label(rects1, padding=3, fmt='%.1f%%', fontsize=10)
    ax.bar_label(rects2, padding=3, fmt='%.1f%%', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=300)
    print("\n✅ Saved comparison chart to 'baseline_comparison.png'")
    plt.show()

if __name__ == "__main__":
    main()