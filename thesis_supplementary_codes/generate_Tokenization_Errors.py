import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
from tqdm import tqdm
import os

# 1. LOAD MODEL & TEST DATA
MODEL_PATH = os.path.join('..', 'outputs', 'fast_model_standard')
DATA_ROOT = os.path.join('..', 'data')
TEST_FILE = os.path.join(DATA_ROOT, 'test', 'test.tsv')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
df = pd.read_csv(TEST_FILE, sep='\t')
df['image_path'] = df['image'].apply(lambda x: os.path.join(DATA_ROOT, 'test', 'images', str(x)))

# 2. RUN INFERENCE
predictions = []
references = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    try:
        image = Image.open(row['image_path']).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=128, num_beams=5)
        pred_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(pred_text)
        references.append(row['text'])
    except Exception as e:
        predictions.append('')
        references.append(row['text'])

# 3. TOKENIZATION ERROR ANALYSIS (N=5)
def analyze_tokenization_errors(preds, refs, N=5):
    errors = []
    for gt, pred in zip(refs, preds):
        if gt.strip() != pred.strip():
            # Simple heuristic: missing chars, first char after space, etc.
            gt_tokens = gt.split()
            pred_tokens = pred.split()
            missing = [t for t in gt_tokens if t not in pred_tokens]
            error_desc = []
            if missing:
                error_desc.append(f"Missing: {' '.join(missing)}")
            if len(gt_tokens) > 0 and len(pred_tokens) > 0 and gt_tokens[0] != pred_tokens[0]:
                error_desc.append("First char after space mismatch")
            if not error_desc:
                error_desc.append("Tokenization mismatch")
            errors.append([gt, pred, '; '.join(error_desc)])
        if len(errors) >= N:
            break
    return errors

data = analyze_tokenization_errors(predictions, references, N=5)
columns = ["Ground Truth (GT)", "Prediction (Standard Loss)", "Error Analysis"]

# 4. SETUP PLOT
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# 5. CONFIGURE FONT 
# Windows usually has 'Nyala'. If on Linux/Mac, use 'Noto Sans Ethiopic' or 'Abyssinica SIL'
font_path = "C:/Windows/Fonts/Nyala.ttf" 
try:
    prop = font_manager.FontProperties(fname=font_path)
    print(f"✅ Loaded font: {prop.get_name()}")
except:
    print("⚠️ Font not found. Using default (Ethiopic might not render).")
    prop = None

# 6. CREATE TABLE
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc='center',
    cellLoc='left',
    colWidths=[0.3, 0.3, 0.4]
)

# 7. STYLE TABLE
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2) # Adjust row height

# Style Headers
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#40466e') # Academic Blue header
        cell.set_fontsize(12)
    else:
        # Set Ethiopic font for data rows
        cell.set_text_props(fontproperties=prop)
        # Highlight errors in the prediction column (Column 1)
        if col == 1:
            cell.set_text_props(color='#c0392b', fontproperties=prop) # Red text for errors
        
        # Alternate row colors
        if row % 2 == 0:
            cell.set_facecolor('#f5f5f5')

# 8. SAVE
plt.title("Figure 3.4: Systematic Tokenization Errors (Character Elision) Before Custom Weighting", 
          fontsize=14, pad=20, weight='bold')
plt.savefig("tokenizer_error_analysis.png", dpi=300, bbox_inches='tight')
print("✅ Image saved as 'tokenizer_error_analysis.png'")
plt.show()