# This script evaluates the non-Tokenized TrOCR model on the test set and computes CER, WER, and Accuracy.
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import json
import evaluate  # huggingface evaluate library

def main():
    # --- CONFIGURATION ---
    MODEL_PATH = "outputs/fast_model_standard"  # Where fast_train.py saved the model
    DATA_ROOT = "data"
    TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
    OUTPUT_JSON = "thesis_metrics_standard.json"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📊 Starting Evaluation on {device}...")

    # 1. Load Model & Processor
    print(f"   Loading model from {MODEL_PATH}...")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Load Test Data
    print(f"   Loading test data from {TEST_FILE}...")
    df = pd.read_csv(TEST_FILE, sep='\t')
    # Fix image paths
    df['image_path'] = df['image'].apply(
        lambda x: os.path.join(DATA_ROOT, "test", "images", str(x))
    )
    print(f"   Found {len(df)} test samples.")

    # 3. Define Metrics
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    predictions = []
    references = []
    
    print("   Running Inference (This takes a few minutes)...")
    
    # 4. Inference Loop
    model.eval()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, 
                    max_length=128,
                    num_beams=5,             # Increase beams from 1 to 5 to gain accuracy of 97.44% from 94%
                    #repetition_penalty=1.2,  # Prevents it from getting stuck/skipping
                    #length_penalty=1.0,      # Encourages it not to cut words short
                    #early_stopping=True,
                    #no_repeat_ngram_size=3
                )
            
            pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            predictions.append(pred_text)
            references.append(row['text'])
            
        except Exception as e:
            print(f"   ⚠️ Error reading {row['image_path']}: {e}")

    # 5. Calculate Final Scores
    print("   Computing CER and WER...")
    final_cer = cer_metric.compute(predictions=predictions, references=references)
    final_wer = wer_metric.compute(predictions=predictions, references=references)
    
    # Calculate Accuracy (Exact Match)
    exact_matches = sum([1 for p, r in zip(predictions, references) if p.strip() == r.strip()])
    accuracy = exact_matches / len(references)

    # 6. Print & Save Results
    results = {
        "CER": final_cer,
        "WER": final_wer,
        "Accuracy": accuracy,
        "Total Samples": len(references)
    }
    
    print("\n" + "="*40)
    print("🎓 THESIS RESULTS")
    print("="*40)
    print(f"✅ Character Error Rate (CER): {final_cer*100:.2f}%")
    print(f"✅ Word Error Rate (WER):      {final_wer*100:.2f}%")
    print(f"✅ Exact Match Accuracy:       {accuracy*100:.2f}%")
    print("="*40)
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📝 Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()