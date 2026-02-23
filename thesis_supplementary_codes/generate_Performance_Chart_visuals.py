import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import font_manager
from PIL import Image
import os
import json

# --- CONFIGURATION ---
OUTPUT_DIR = "thesis_visuals"
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
    
    # Load latest evaluation metrics
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'thesis_metrics.json')
    if not os.path.exists(metrics_path):
        metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'thesis_metrics.json')
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Could not find thesis_metrics.json. Checked: {metrics_path}")
    with open(metrics_path, "r") as f:
        results = json.load(f)

    # Convert to percentages
    accuracy = results["Accuracy"] * 100
    wer = results["WER"] * 100
    cer = results["CER"] * 100

    metrics = {
        'Exact Match\nAccuracy': round(accuracy, 2),
        'Word Error Rate\n(WER)': round(wer, 2),
        'Char Error Rate\n(CER)': round(cer, 2)
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

    plt.title(f"Final Model Performance on Test Set (N={results['Total Samples']})", fontsize=14, weight='bold', pad=20)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.ylim(0, 110) # Give space for text

    # Save
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figure_performance_metrics.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance()
    print(f"\n✅ Performance metrics visualization saved to '{OUTPUT_DIR}/'")