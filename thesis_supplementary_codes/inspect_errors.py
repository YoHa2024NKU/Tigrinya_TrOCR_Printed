import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import pandas as pd
import os

# --- CONFIG ---
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)

print("Loading test data...")
df = pd.read_csv(TEST_FILE, sep='\t')
df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))

print("\n🔍 INSPECTING FIRST 10 SAMPLES:\n")
print(f"{'IMAGE':<15} | {'GROUND TRUTH':<40} | {'PREDICTION':<40}")
print("-" * 100)

for i in range(10):
    row = df.iloc[i]
    image = Image.open(row['image_path']).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    # Generate with Beam Search
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=128, num_beams=4, early_stopping=True)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    gt_text = row['text']
    img_name = os.path.basename(row['image_path'])
    
    # Mark mismatches
    mark = "✅" if pred_text.strip() == gt_text.strip() else "❌"
    
    print(f"{img_name[:15]:<15} | {gt_text[:38]:<40} | {pred_text[:38]:<40} {mark}")

print("-" * 100)