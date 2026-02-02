import os
from sympy import evaluate
import torch
import pandas as pd
import shutil
import evaluate
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
OUTPUT_DIR = "thesis_difficult_samples"
NUM_SAMPLES_TO_SAVE = 100  # How many "bad" images do you want?

# Ensure output directory exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Finding difficult samples using {device}...")

# 1. Load Model
try:
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# 2. Load Data
df = pd.read_csv(TEST_FILE, sep='\t')
df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))

# Metric calculator
cer_metric = evaluate.load("cer")

results = []

print(f"🔍 Scanning {len(df)} images to find the hardest ones...")

# 3. Inference Loop
for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image_path = row['image_path']
        gt_text = str(row['text'])
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            # Use your BEST generation settings
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=5,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Calculate CER for this specific sample
        # We handle empty strings to prevent division by zero errors
        if len(gt_text) > 0:
            sample_cer = cer_metric.compute(predictions=[pred_text], references=[gt_text])
        else:
            sample_cer = 1.0 if len(pred_text) > 0 else 0.0
            
        # Store result
        results.append({
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "gt": gt_text,
            "pred": pred_text,
            "cer": sample_cer
        })

    except Exception as e:
        print(f"Skipping {row['image']}: {e}")

# 4. Sort by Error Rate (Highest CER first)
# We filter out CER > 1.0 (sometimes happens with very short text) to keep realistic samples
sorted_results = sorted(results, key=lambda x: x['cer'], reverse=True)

print(f"\n📸 Saving top {NUM_SAMPLES_TO_SAVE} difficult images to '{OUTPUT_DIR}/'...")

report_lines = []
report_lines.append("THESIS ERROR ANALYSIS REPORT")
report_lines.append("============================")

for i in range(min(NUM_SAMPLES_TO_SAVE, len(sorted_results))):
    item = sorted_results[i]
    
    # Copy image to output folder
    # We rename it to include the rank (e.g., "1_filename.jpg") for easy sorting
    new_filename = f"{i+1}_{item['filename']}"
    dest_path = os.path.join(OUTPUT_DIR, new_filename)
    shutil.copy(item['image_path'], dest_path)
    
    # Add to report
    error_percent = item['cer'] * 100
    report_lines.append(f"\n#{i+1} Filename: {item['filename']}")
    report_lines.append(f"   Error Rate: {error_percent:.2f}%")
    report_lines.append(f"   Ground Truth: {item['gt']}")
    report_lines.append(f"   Prediction:   {item['pred']}")
    report_lines.append("-" * 50)

# Save Report
with open(os.path.join(OUTPUT_DIR, "error_analysis_report.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("✅ Done! Check the folder 'thesis_difficult_samples' for images and the report.")