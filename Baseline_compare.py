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
HANDWRITTEN_FINETUNED = "outputs/fast_model/best_model"
PRINTED_FINETUNED = "outputs/fast_model_printed/best_model"
VANILLA_HANDWRITTEN = "microsoft/trocr-base-handwritten"
VANILLA_PRINTED = "microsoft/trocr-base-printed"

DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 500

def evaluate_model(model_name_or_path, df, desc):
    print(f"\n🔄 Loading {desc} ({model_name_or_path})...")
    try:
        processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path).to(DEVICE)
    except Exception as e:
        print(f"❌ Error loading {model_name_or_path}: {e}")
        return 1.0, 1.0, 0.0

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    preds, refs = [],[]

    print(f"   Running Inference on {len(df)} samples...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

            with torch.no_grad():
                # Explicitly use keyword argument pixel_values=pixel_values to avoid warnings
                generated_ids = model.generate(pixel_values=pixel_values, max_length=128, num_beams=5)

            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            preds.append(pred)
            
            # Cast to string safely to avoid NaN float errors
            refs.append(str(row['text']) if pd.notna(row['text']) else "")
        except Exception as e:
            continue
            
    # Prevent ZeroDivisionError if all images failed to load
    if len(refs) == 0:
        return 1.0, 1.0, 0.0

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
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)

    # 2. Evaluate All Four Models
    vh_cer, vh_wer, vh_acc = evaluate_model(VANILLA_HANDWRITTEN, df, "Vanilla Handwritten")
    vp_cer, vp_wer, vp_acc = evaluate_model(VANILLA_PRINTED, df, "Vanilla Printed")
    fh_cer, fh_wer, fh_acc = evaluate_model(HANDWRITTEN_FINETUNED, df, "Fine-tuned Handwritten")
    fp_cer, fp_wer, fp_acc = evaluate_model(PRINTED_FINETUNED, df, "Fine-tuned Printed")

    # 3. Print Table
    print("\n" + "=" * 90)
    print(f"📊 BASELINE COMPARISON (N={SAMPLE_SIZE})")
    print("=" * 90)
    print(f"{'Model':<35} | {'CER':<10} | {'WER':<10} | {'Accuracy':<10}")
    print("-" * 90)
    print(f"{'TrOCR-base-handwritten (zero-shot)':<35} | {vh_cer*100:6.2f}%   | {vh_wer*100:6.2f}%   | {vh_acc*100:6.2f}%")
    print(f"{'TrOCR-base-printed (zero-shot)':<35} | {vp_cer*100:6.2f}%   | {vp_wer*100:6.2f}%   | {vp_acc*100:6.2f}%")
    print("-" * 90)
    print(f"{'TrOCR-base-handwritten (fine-tuned)':<35} | {fh_cer*100:6.2f}%   | {fh_wer*100:6.2f}%   | {fh_acc*100:6.2f}%")
    print(f"{'TrOCR-base-printed (fine-tuned)':<35} | {fp_cer*100:6.2f}%   | {fp_wer*100:6.2f}%   | {fp_acc*100:6.2f}%")
    print("=" * 90)

    # 4. Generate Chart
    labels =['CER', 'WER', 'Accuracy']

    vanilla_hw =[vh_cer * 100, vh_wer * 100, vh_acc * 100]
    vanilla_pr =[vp_cer * 100, vp_wer * 100, vp_acc * 100]
    finetuned_hw =[fh_cer * 100, fh_wer * 100, fh_acc * 100]
    finetuned_pr =[fp_cer * 100, fp_wer * 100, fp_acc * 100]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 8))

    r1 = ax.bar(x - 1.5 * width, vanilla_hw, width, label='Handwritten (zero-shot)', color='#e74c3c', alpha=0.8)
    r2 = ax.bar(x - 0.5 * width, vanilla_pr, width, label='Printed (zero-shot)', color='#e67e22', alpha=0.8)
    r3 = ax.bar(x + 0.5 * width, finetuned_hw, width, label='Handwritten (fine-tuned)', color='#2ecc71', alpha=0.9)
    r4 = ax.bar(x + 1.5 * width, finetuned_pr, width, label='Printed (fine-tuned)', color='#3498db', alpha=0.9)

    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: Zero-Shot vs Fine-Tuned (N=500)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    
    # --- CHANGED THIS LINE (loc='upper left' -> loc='upper right') ---
    ax.legend(fontsize=10, loc='upper right')
    # -----------------------------------------------------------------
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)

    ax.bar_label(r1, padding=3, fmt='%.1f%%', fontsize=8)
    ax.bar_label(r2, padding=3, fmt='%.1f%%', fontsize=8)
    ax.bar_label(r3, padding=3, fmt='%.1f%%', fontsize=8, weight='bold')
    ax.bar_label(r4, padding=3, fmt='%.1f%%', fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig("baseline_comparison1.png", dpi=300)
    print("\n✅ Saved comparison chart to 'baseline_comparison1.png'")
    plt.show()

if __name__ == "__main__":
    main()