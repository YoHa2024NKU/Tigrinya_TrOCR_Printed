import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, avoids GUI hang

import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# --- CONFIGURATION ---
OUTPUT_DIR = r"thesis_visuals"
METRICS_PATH = r"D:\\Tigrinya_OCR_Project\\thesis_metrics.json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Set seaborn style
sns.set_theme(style="whitegrid", font_scale=1.2)


def plot_performance():
    print("Generating Performance Chart...")

    if not os.path.exists(METRICS_PATH):
        raise FileNotFoundError(f"Could not find thesis_metrics.json at: {METRICS_PATH}")

    with open(METRICS_PATH, "r") as f:
        results = json.load(f)

    # Convert to percentages
    accuracy = round(results["Accuracy"] * 100, 2)
    wer = round(results["WER"] * 100, 2)
    cer = round(results["CER"] * 100, 2)

    metrics = ['Exact Match\nAccuracy', 'Word Error Rate\n(WER)', 'Char Error Rate\n(CER)']
    values = [accuracy, wer, cer]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = sns.barplot(x=metrics, y=values, palette=colors, edgecolor='black', ax=ax)

    # Add values on top of bars
    for i, (val, bar) in enumerate(zip(values, ax.patches)):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontsize=12, weight='bold')

    ax.set_title(f"Final Model Performance on Test Set (N={results['Total Samples']})",
                 fontsize=14, weight='bold', pad=20)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0, 110)

    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure_performance_metrics.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    # Open the saved image
    try:
        os.startfile(output_path)
    except AttributeError:
        pass

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_performance()
    print(f"\nPerformance metrics visualization saved to '{OUTPUT_DIR}/'")