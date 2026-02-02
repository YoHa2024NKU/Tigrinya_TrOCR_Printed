import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import font_manager
from PIL import Image
import os

# --- CONFIGURATION ---
OUTPUT_DIR = "thesis_visuals"
SAMPLE_IMG_DIR = "thesis_difficult_samples"  # From previous step
FONT_PATH = "C:/Windows/Fonts/Nyala.ttf"     # Standard Windows Ethiopic Font

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. SETUP FONT
try:
    prop = font_manager.FontProperties(fname=FONT_PATH)
    print(f"✅ Loaded font: {prop.get_name()}")
except:
    print("⚠️ Font not found. Visualizations might show boxes for Ethiopic text.")
    prop = None

# =============================================================================
# 1. PERFORMANCE METRICS (Bar Chart)
# =============================================================================
def plot_performance():
    print("📊 Generating Performance Chart...")
    
    # Data from your final evaluation
    metrics = {
        'Exact Match\nAccuracy': 94.42,
        'Word Error Rate\n(WER)': 1.66,
        'Char Error Rate\n(CER)': 0.41
    }
    
    names = list(metrics.keys())
    values = list(metrics.values())
    colors = ['#2ecc71', '#f1c40f', '#e74c3c'] # Green, Yellow, Red

    plt.figure(figsize=(9, 6))
    bars = plt.bar(names, values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}%', ha='center', va='bottom', fontsize=12, weight='bold')

    plt.title("Final Model Performance on Test Set (N=5,000)", fontsize=14, weight='bold', pad=20)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.ylim(0, 110) # Give space for text
    
    # Save
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_performance_metrics.png", dpi=300)
    plt.close()

# =============================================================================
# 2. ERROR DISTRIBUTION (Rendered Table)
# =============================================================================
def plot_error_table():
    print("📊 Generating Error Distribution Table...")
    
    # Based on the categories you provided
    data = [
        ["Mixed Script", 5, "50%"],
        ["Numeric Errors", 5, "50%"],
        ["Total", 10, "100%"]
    ]
    
    columns = ["Error Category", "Frequency", "Percentage"]
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#2c3e50']*3
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)
    
    # Styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
        elif row == len(data): # Total row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#ecf0f1')
        
    plt.title("Table 4.5: Error Distribution by Category (Sample N=10)", fontsize=14, weight='bold', pad=10)
    plt.savefig(f"{OUTPUT_DIR}/table_error_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 3. SAMPLE ERROR CASES (Visuals)
# =============================================================================
def plot_visual_errors():
    print("📸 Generating Visual Error Cases...")
    
    # Using specific data from  previous error report
    # We select 3 examples that look like "Diacritic/Visual" confusion
    samples = [
        {
            "filename": "2_03001.jpg", # Assuming this exists in the difficult samples folder
            "gt": "ን22/1/18 ኣብ ቤ/ፍ",
            "pred": "ን27/11/17 ኣብ ቤ/ፍ",
            "desc": "Number Confusion: '2' (7) vs '8' (7)"
        },
        {
            "filename": "9_02974.jpg", 
            "gt": "ን26/1/18 ኣብ ቤ/ፍ",
            "pred": "ን25/11/18 ኣብ ቤ/ፍ",
            "desc": "Number Confusion: '6' (5) vs '/' (1)"
        },
        {
            "filename": "10_03018.jpg",
            "gt": "ን22/1/18 ኣብ ቤ/ፍ",
            "pred": "ን223/11/18 ኣብ ቤ/ፍ",
            "desc": "Cross-Script: '46' vs 'ፋዕ'"
        }
    ]

    fig, axes = plt.subplots(len(samples), 1, figsize=(8, 6))
    plt.subplots_adjust(hspace=0.4)
    
    for i, sample in enumerate(samples):
        ax = axes[i]
        img_path = os.path.join(SAMPLE_IMG_DIR, sample["filename"])
        
        # Load Image (Create dummy if missing for testing)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            ax.imshow(img)
        else:
            # Placeholder if file doesn't exist
            ax.text(0.5, 0.5, f"Image {sample['filename']} not found", ha='center')
            
        ax.axis('off')
        
        # Text Info
        # ax.set_title(f"Case {i+1}: {sample['desc']}", fontsize=12, weight='bold', loc='left')
        
        # GT and Pred Text
        # GT
        ax.text(0, -0.15, f"GT:   {sample['gt']}", color='#27ae60', 
                transform=ax.transAxes, fontsize=20, fontproperties=prop, weight='bold')
        # Pred
        ax.text(0, -0.40, f"Pred: {sample['pred']}", color='#c0392b', 
                transform=ax.transAxes, fontsize=20, fontproperties=prop, weight='bold')

    plt.suptitle("Figure 4.6: Sample Error Cases (Diacritic & Visual Confusion)", fontsize=16, weight='bold')
    plt.savefig(f"{OUTPUT_DIR}/figure_error_cases.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_performance()
    plot_error_table()
    plot_visual_errors()
    print(f"\n✅ All visualizations saved to '{OUTPUT_DIR}/'")