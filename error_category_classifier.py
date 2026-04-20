import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import numpy as np

# --- CONFIGURATION ---
MODEL_PATH = "outputs/fast_model_printed/best_model"
DATA_ROOT = "data"
TEST_FILE = os.path.join(DATA_ROOT, "test", "test.tsv")
OUTPUT_DIR = "thesis_visuals"
SAMPLE_SIZE = 5000
FONT_PATH = "C:/Windows/Fonts/nyala.ttf" 

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Ge'ez LINGUISTIC MAPS (304+ Characters) ---
FAMILIES = [
    "ሀሁሂሃሄህሆ", "ለሉሊላሌልሎሏ", "ሐሑሒሓሔሕሖሗ", "መሙሚማሜምሞሟ", "ሠሡሢሣሤሥሦሧ", 
    "ረሩሪራሬርሮሯ", "ሰሱሲሳሴስሶሷ", "ሸሹሺሻሼሽሾሿ", "ቀቁቂቃቄቅቆቈቊቋቌቍ", "በቡቢባቤብቦቧ", 
    "ቨቩቪቫቬቭቮቯ", "ተቱቲታቴትቶቷ", "ቸቹቺቻቼችቾቿ", "ኀኁኂኃኄኅኆኈኊኋኌኍ", "ነኑኒናኔንኖኗ", 
    "ኘኙኚኛኜኝኞኟ", "አኡኢኣኤእኦኧ", "ከኩኪካኬክኮኰኲኳኴኵ", "ኸኹኺኻኼኽኾዀዂዃዄዅ", 
    "ወዉዊዋዌውዎ", "ዐዑዒዓዔዕዖ", "ዘዙዚዛዜዝዞዟ", "ዠዡዢዣዤዥዦዧ", "የዩዪያዬይዮ", 
    "ደዱዲዳዴድዶዷ", "ጀጁጂጃጄጅጆጇ", "ገጉጊጋጌግጎጐጒጓኄጕ", "ጠጡጢጣጤጥጦጧ", 
    "ጨጩጪጫጬጭጮጯ", "ጰ኱ጲጳጴጵጶጷ", "ጸጹጺጻጼጽጾጿ", "ፀፁፂፃፄፅፆ", "ፈፉፊፋፌፍፎፏ", "ፐፑፒፓፔፕፖፗ"
]
char_to_family = {char: i for i, fam in enumerate(FAMILIES) for char in fam}
LABIALIZED = set("ሏሗሟሧሯሷሿቧቯቷቿኗኟኧዟዧዷጇጧጯጿፏፗቈቊቋቌቍኰኲኳኴኵጐጒጓኄጕኈኊኋኌኍዀዂዃዄዅ")
SIMILAR_PAIRS = [('ሀ', 'ሃ'), ('ለ', 'ረ'), ('ሰ', 'ሸ'), ('በ', 'ቨ'), ('ተ', 'ቸ'), ('ነ', 'ኘ'), ('አ', 'ዐ'), ('ከ', 'ኸ'), ('ወ', 'ዐ'), ('ደ', 'ጀ'), ('ገ', 'ጎ'), ('ጠ', 'ጨ'), ('ጸ', 'ፀ'), ('ፈ', 'ፐ')]

def classify_error(gt, pred):
    gt, pred = str(gt).strip(), str(pred).strip()
    if gt == pred: return "Correct Prediction"
    if re.search(r'[0-9a-zA-Z፩-፼]', gt) or re.search(r'[0-9a-zA-Z፩-፼]', pred):
        if re.sub(r'[^0-9a-zA-Z፩-፼]', '', gt) != re.sub(r'[^0-9a-zA-Z፩-፼]', '', pred): return "Numbers and Mixed-Script Error"
    if gt.replace(" ", "") == pred.replace(" ", ""): return "Boundary and Spacing Error"
    if re.sub(r'[^\w\s]', '', gt) == re.sub(r'[^\w\s]', '', pred): return "Punctuation Error"
    if (len(pred) < len(gt) * 0.7) or (pred in gt and len(pred) < len(gt)): return "Partial Recognition"
    if len(gt) == len(pred):
        is_diacritic, is_visual_sub, is_labial = True, True, False
        for g, p in zip(gt, pred):
            if g != p:
                if g in LABIALIZED or p in LABIALIZED: is_labial = True
                if not (g in char_to_family and p in char_to_family and char_to_family[g] == char_to_family[p]): is_diacritic = False
                if not any((g == s1 and p == s2) or (g == s2 and p == s1) for s1, s2 in SIMILAR_PAIRS): is_visual_sub = False
        if is_labial: return "Labialized Character Error"
        if is_diacritic: return "Diacritic Confusion"
        if is_visual_sub: return "Character Substitution (Visual)"
    return "Character Substitution (Visual)"

def main():
    print(f"🚀 Processing {SAMPLE_SIZE} samples...")
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(device)
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    df = pd.read_csv(TEST_FILE, sep='\t')
    df_test = df.head(SAMPLE_SIZE).copy()
    df_test['image_path'] = df_test['image'].apply(lambda x: os.path.join(DATA_ROOT, "test", "images", str(x)))
    
    cats = ["Diacritic Confusion", "Labialized Character Error", "Boundary and Spacing Error", "Punctuation Error", "Numbers and Mixed-Script Error", "Partial Recognition", "Character Substitution (Visual)"]
    gallery_samples = {cat: [] for cat in cats}
    all_results = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        try:
            image_pil = Image.open(row['image_path']).convert("RGB")
            pixel_values = processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                generated_ids = model.generate(pixel_values, max_length=128, num_beams=5)
            pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            gt, category = row['text'], classify_error(row['text'], pred)
            all_results.append(category)
            if category in gallery_samples and len(gallery_samples[category]) < 2:
                gallery_samples[category].append({"image": image_pil, "gt": gt, "pred": pred})
        except: continue

    # ==========================================
    # 1. FINAL COMPACT GALLERY (STYLIZED & HEIGHT-FIXED)
    # ==========================================
    valid_cats = [c for c in cats if len(gallery_samples[c]) > 0]
    if valid_cats:
        print("🎨 Drawing Final Stylized Gallery...")
        # USER SETTINGS INCORPORATED:
        card_w, card_h, pad = 420, 220, 25 
        try:
            f_main = ImageFont.truetype(FONT_PATH, 28) 
            f_bold = ImageFont.truetype(FONT_PATH, 36) 
            f_cat = ImageFont.truetype(FONT_PATH, 24)  
        except: f_main = f_bold = f_cat = ImageFont.load_default()

        # FIXED HEIGHT MATH: Top Header (120) + Rows * (Card + Pad) + Buffer (50)
        canvas_h = 120 + (len(valid_cats) * (card_h + pad)) + 50
        canvas_w = (card_w * 2) + (pad * 3)
        canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((canvas_w//2 - 350, 30), "Figure 4.6: Qualitative Analysis of Failure Cases", fill=(0,0,0), font=f_bold)

        curr_y = 120
        for cat in valid_cats:
            curr_x = pad
            samples = gallery_samples[cat]
            for i in range(2):
                if i < len(samples):
                    sample = samples[i]
                    draw.rectangle([curr_x, curr_y, curr_x+card_w, curr_y+card_h], outline=(215,215,215), width=1)
                    draw.text((curr_x + 20, curr_y + 12), f"Category: {cat}", fill=(0, 51, 102), font=f_cat)
                    
                    img = sample['image'].copy()
                    if img.width > (card_w - 40): img.thumbnail((card_w - 40, 150))
                    canvas.paste(img, (curr_x + 20, curr_y + 60))
                    
                    # USER COLORS & SPACING INCORPORATED:
                    text_y = curr_y + 60 + img.height + 8
                    draw.text((curr_x + 20, text_y), f"GT: {sample['gt']}", fill=(55,125,34), font=f_main)
                    draw.text((curr_x + 20, text_y + 38), f"Pred: {sample['pred']}", fill=(179,39,27), font=f_main)
                curr_x += card_w + pad
            curr_y += card_h + pad
        canvas.save(os.path.join(OUTPUT_DIR, "error_sample_gallery_PIL.png"))

    # ==========================================
    # 2. STATS TABLE
    # ==========================================
    stats_df = pd.Series(all_results).value_counts().reset_index()
    stats_df.columns = ['Error Category', 'Count']
    stats_df['Percentage (%)'] = (stats_df['Count'] / len(all_results)) * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight'); ax.axis('off')
    t_data = [[r['Error Category'], int(r['Count']), f"{r['Percentage (%)']:.2f}%"] for _, r in stats_df.iterrows()]
    t_data.append(["TOTAL SAMPLES", len(all_results), "100.00%"])
    tbl = ax.table(cellText=t_data, colLabels=["Error Category", "Frequency", "Distribution (%)"], loc='center', cellLoc='center', colWidths=[0.5, 0.2, 0.3])
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1.2, 3.2)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0: cell.set_text_props(weight='bold', color='white'); cell.set_facecolor('#2c3e50')
        elif row == len(t_data): cell.set_text_props(weight='bold'); cell.set_facecolor('#ecf0f1')
    plt.title(f"Table 4.4: Qualitative Error Analysis (N={len(all_results)})", fontsize=16, weight='bold', pad=30)
    plt.savefig(os.path.join(OUTPUT_DIR, "final_distribution_table.png"), dpi=300, bbox_inches='tight')
    print("🏁 Done. Gallery and Table saved to 'thesis_visuals'.")

if __name__ == "__main__": main()