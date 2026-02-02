import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import re

# --- CONFIGURATION ---
MODEL_PATH = "outputs/fast_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
OUTPUT_DIR = "thesis_visuals"
SAMPLE_SIZE = 100  # As requested

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ERROR CLASSIFIER LOGIC ---
def classify_error(gt, pred):
    """
    Categorizes the error based on linguistic rules.
    """
    gt = str(gt).strip()
    pred = str(pred).strip()
    
    # 1. No Error
    if gt == pred:
        return "Correct Prediction"

    # 2. Numerical / Mixed Script (The major weakness we found)
    # Check if GT contains digits (0-9) and Pred doesn't match
    if re.search(r'\d', gt):
        # If numbers are completely missing or wrong
        gt_nums = re.findall(r'\d+', gt)
        pred_nums = re.findall(r'\d+', pred)
        if gt_nums != pred_nums:
            return "Numerical/Mixed-Script Error"

    # 3. Punctuation / Spacing
    # Remove all punctuation and spaces, check if letters match
    def clean_text(text):
        return re.sub(r'[^\w]', '', text)
    
    if clean_text(gt) == clean_text(pred):
        return "Punctuation/Spacing Error"

    # 4. Character Elision (The 'Eating Letters' bug)
    # Check if Pred is a substring of GT (meaning chars are missing)
    if pred in gt and len(pred) < len(gt):
        return "Partial Recognition (Elision)"

    # 5. Length Mismatch (Hallucination or severe cut-off)
    if abs(len(gt) - len(pred)) > 3:
        return "Severe Length Mismatch"

    # 6. Default: Visual/Diacritic Confusion
    # (If lengths are similar but chars are different)
    return "Visual/Diacritic Confusion"

# --- MAIN SCRIPT ---
def main():
    print(f"🚀 Starting Error Analysis on {SAMPLE_SIZE} samples...")
    
    # Load Model
    try:
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
        processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Load Data
    df = pd.read_csv(TEST_FILE, sep='\t')
    df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))
    
    # Pick Random 100 Samples
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    
    results = []
    
    print("🔍 Running Inference & Classification...")
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        try:
            image = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values, max_length=128, num_beams=5, 
                    repetition_penalty=1.2, no_repeat_ngram_size=3, early_stopping=True
                )
            
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            gt = row['text']
            
            category = classify_error(gt, pred)
            results.append(category)
            
        except:
            continue

    # --- STATISTICS ---
    # Count frequencies
    counts = pd.Series(results).value_counts()
    total_errors = sum(counts) - counts.get("Correct Prediction", 0)
    
    # Create DataFrame for Table
    df_stats = pd.DataFrame({
        "Error Category": counts.index,
        "Frequency": counts.values,
        "Percentage (%)": [(x / SAMPLE_SIZE) * 100 for x in counts.values]
    })
    
    # Filter out "Correct Prediction" to focus only on errors for the thesis table
    df_errors = df_stats[df_stats["Error Category"] != "Correct Prediction"].copy()
    
    # Recalculate percentage relative to Total Errors (optional, but good for error breakdown)
    # Or keep it relative to Total Samples (as requested). Let's stick to Total Samples.
    
    print("\n" + "="*50)
    print("📊 ERROR DISTRIBUTION TABLE (N=100)")
    print("="*50)
    print(df_stats.to_string(index=False))
    print("="*50)
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "error_distribution_table.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"📝 CSV saved to {csv_path}")

    # --- GENERATE IMAGE TABLE (FOR THESIS) ---
    fig, ax = plt.subplots(figsize=(8, len(df_stats) * 0.6 + 1)) # Dynamic height
    ax.axis('tight')
    ax.axis('off')
    
    # Format data for table
    table_data = []
    for _, row in df_stats.iterrows():
        table_data.append([row['Error Category'], row['Frequency'], f"{row['Percentage (%)']:.1f}%"])
    
    # Add Total Row
    table_data.append(["TOTAL", SAMPLE_SIZE, "100%"])

    table = ax.table(
        cellText=table_data,
        colLabels=["Error Category", "Frequency", "Percentage"],
        loc='center',
        cellLoc='center',
        colWidths=[0.5, 0.2, 0.3]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2c3e50')
        elif row == len(table_data): # Total Row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#bdc3c7')
        elif table_data[row-1][0] == "Correct Prediction": # Correct Row
            cell.set_facecolor('#d5f5e3') # Light Green
            
    plt.title(f"Table 4.5: Error Distribution Analysis (Sample N={SAMPLE_SIZE})", fontsize=14, weight='bold', pad=20)
    
    img_path = os.path.join(OUTPUT_DIR, "table_error_distribution_100.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"📸 Image saved to {img_path}")

if __name__ == "__main__":
    main()