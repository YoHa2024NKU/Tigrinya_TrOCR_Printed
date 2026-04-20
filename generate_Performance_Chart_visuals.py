import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# --- CONFIGURATION ---
OUTPUT_DIR = r"thesis_visuals"
HW_METRICS_PATH = r"D:\Tigrinya_OCR_Project\thesis_metrics_handwritten.json"
PR_METRICS_PATH = r"D:\Tigrinya_OCR_Project\thesis_metrics_printedd.json"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

sns.set_theme(style="whitegrid", font_scale=1.2)


def plot_performance():
    print("Generating Performance Chart...")

    if not os.path.exists(HW_METRICS_PATH):
        raise FileNotFoundError(f"Could not find: {HW_METRICS_PATH}")
    if not os.path.exists(PR_METRICS_PATH):
        raise FileNotFoundError(f"Could not find: {PR_METRICS_PATH}")

    with open(HW_METRICS_PATH, "r") as f:
        hw = json.load(f)
    with open(PR_METRICS_PATH, "r") as f:
        pr = json.load(f)

    # Handwritten variant
    hw_accuracy = round(hw["Accuracy"] * 100, 2)
    hw_wer = round(hw["WER"] * 100, 2)
    hw_cer = round(hw["CER"] * 100, 2)

    # Printed variant
    pr_accuracy = round(pr["Accuracy"] * 100, 2)
    pr_wer = round(pr["WER"] * 100, 2)
    pr_cer = round(pr["CER"] * 100, 2)

    total_samples = hw.get("Total Samples", 5000)

    # --- Data setup ---
    metrics = [
        'Exact Match\nAccuracy',
        'Word Error Rate\n(WER)',
        'Char Error Rate\n(CER)',
    ]

    hw_values = [hw_accuracy, hw_wer, hw_cer]
    pr_values = [pr_accuracy, pr_wer, pr_cer]

    x = np.arange(len(metrics))
    bar_width = 0.32

    hw_color = '#3498db'   # blue for handwritten
    pr_color = '#e67e22'   # orange for printed

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_hw = ax.bar(
        x - bar_width / 2, hw_values, bar_width,
        label='Handwritten variant', color=hw_color, edgecolor='black'
    )
    bars_pr = ax.bar(
        x + bar_width / 2, pr_values, bar_width,
        label='Printed variant', color=pr_color, edgecolor='black'
    )

    for bar in bars_hw:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + 0.8,
            f'{height}%', ha='center', va='bottom',
            fontsize=11, weight='bold', color=hw_color,
        )

    for bar in bars_pr:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + 0.8,
            f'{height}%', ha='center', va='bottom',
            fontsize=11, weight='bold', color=pr_color,
        )

    ax.set_title(
        f"Fine-Tuned Model Performance on Test Set (N={total_samples})",
        fontsize=14, weight='bold', pad=20,
    )
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=11)

    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "figure_performance_metrics.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    try:
        os.startfile(output_path)
    except AttributeError:
        pass

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_performance()
    print(f"\nPerformance metrics visualization saved to '{OUTPUT_DIR}/'")