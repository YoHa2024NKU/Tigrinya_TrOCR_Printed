import os
import glob
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.font_manager as font_manager

THESIS_VISUALS_DIR = "thesis_visuals"
if not os.path.exists(THESIS_VISUALS_DIR):
    os.makedirs(THESIS_VISUALS_DIR)

# CONFIG
SEARCH_DIR = "outputs"  # We will search inside here
OUTPUT_IMAGE = "training_loss_curve.png"

def find_trainer_state(root_dir):
    print(f"🔍 Searching for 'trainer_state.json' in {root_dir}...")
    # Recursive search for the file
    files = glob.glob(os.path.join(root_dir, "**", "trainer_state.json"), recursive=True)
    
    if not files:
        return None
    
    # If multiple found, take the one with the latest modification time (most recent run)
    latest_file = max(files, key=os.path.getmtime)
    print(f"✅ Found log file: {latest_file}")
    return latest_file

NYALA_FONT_PATH = "nyala.ttf"
nyala_prop = font_manager.FontProperties(fname=NYALA_FONT_PATH)

# Utility: Load predictions and references
def load_predictions_refs():
    # Try to load from thesis_metrics.json or bootstrap_predictions_cache.json
    pred_file = "bootstrap_predictions_cache.json"
    if os.path.exists(pred_file):
        with open(pred_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return cache["preds"], cache["refs"]
    else:
        # Fallback: not available
        print("⚠️ No predictions cache found. Some plots will be skipped.")
        return None, None

# 1. Training Accuracy Curve
def plot_training_accuracy():
    log_file = find_trainer_state(SEARCH_DIR)
    if not log_file:
        print("❌ Error: Could not find 'trainer_state.json'. Skipping accuracy curve.")
        return
    with open(log_file, 'r') as f:
        data = json.load(f)
    history = data.get('log_history', [])
    steps = []
    accs = []
    for entry in history:
        if 'accuracy' in entry and 'step' in entry:
            steps.append(entry['step'])
            accs.append(entry['accuracy']*100)
    if not steps or not accs:
        print("❌ No accuracy data found in logs.")
        return
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    plt.plot(steps, accs, color='#27ae60', linewidth=2.5, label='Training Accuracy')
    plt.title("Figure: Training Accuracy Convergence", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Training Steps", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("Accuracy (%)", fontsize=14, fontproperties=nyala_prop)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{THESIS_VISUALS_DIR}/training_accuracy_curve.png", dpi=300)
    plt.close()
    print(f"✅ Training accuracy curve saved as '{THESIS_VISUALS_DIR}/training_accuracy_curve.png'")

# 2. Confusion Matrix for Fidel Characters
def plot_confusion_matrix():
    preds, refs = load_predictions_refs()
    if preds is None or refs is None:
        return
    # Align character pairs for each prediction-reference
    pred_chars = []
    ref_chars = []
    for p, r in zip(preds, refs):
        min_len = min(len(p), len(r))
        pred_chars.extend(list(p[:min_len]))
        ref_chars.extend(list(r[:min_len]))
    labels = sorted(list(set(ref_chars + pred_chars)))
    cm = confusion_matrix(ref_chars, pred_chars, labels=labels)
    plt.figure(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation='vertical', colorbar=True)
    plt.title("Figure: Fidel Character Confusion Matrix", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Predicted Character", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("True Character", fontsize=14, fontproperties=nyala_prop)
    plt.xticks(fontproperties=nyala_prop)
    plt.yticks(fontproperties=nyala_prop)
    plt.tight_layout()
    plt.savefig(f"{THESIS_VISUALS_DIR}/confusion_matrix_fidel.png", dpi=300)
    plt.close()
    print(f"✅ Confusion matrix saved as '{THESIS_VISUALS_DIR}/confusion_matrix_fidel.png'")

# 3. WER vs. Character Count Breakdown
def plot_wer_vs_char_count():
    preds, refs = load_predictions_refs()
    if preds is None or refs is None:
        return
    from evaluate import load
    wer_metric = load("wer")
    word_lengths = [len(r.split()) for r in refs]
    wers = [wer_metric.compute(predictions=[p], references=[r]) for p, r in zip(preds, refs)]
    plt.figure(figsize=(10, 6))
    plt.scatter(word_lengths, wers, alpha=0.5, color='#e67e22')
    plt.title("Figure: WER vs. Word Count", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Word Count", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("Word Error Rate (WER)", fontsize=14, fontproperties=nyala_prop)
    plt.tight_layout()
    plt.savefig(f"{THESIS_VISUALS_DIR}/wer_vs_word_count.png", dpi=300)
    plt.close()
    print(f"✅ WER vs. word count plot saved as '{THESIS_VISUALS_DIR}/wer_vs_word_count.png'")

# 4. CER Distribution Histogram
def plot_cer_distribution():
    preds, refs = load_predictions_refs()
    if preds is None or refs is None:
        return
    from evaluate import load
    cer_metric = load("cer")
    cers = [cer_metric.compute(predictions=[p], references=[r]) for p, r in zip(preds, refs)]
    plt.figure(figsize=(10, 6))
    sns.histplot(cers, bins=30, color='#2980b9', kde=True)
    plt.title("Figure: CER Distribution Across Test Set", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Character Error Rate (CER)", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("Frequency", fontsize=14, fontproperties=nyala_prop)
    plt.tight_layout()
    plt.savefig(f"{THESIS_VISUALS_DIR}/cer_distribution_histogram.png", dpi=300)
    plt.close()
    print(f"✅ CER distribution histogram saved as '{THESIS_VISUALS_DIR}/cer_distribution_histogram.png'")

# 5. Word Length vs. Accuracy Bar Chart
def plot_word_length_vs_accuracy():
    preds, refs = load_predictions_refs()
    if preds is None or refs is None:
        return
    word_lengths = [len(r.split()) for r in refs]
    accuracies = [int(p.strip() == r.strip()) for p, r in zip(preds, refs)]
    df = {}
    for wl, acc in zip(word_lengths, accuracies):
        df.setdefault(wl, []).append(acc)
    avg_acc = {wl: np.mean(accs)*100 for wl, accs in df.items()}
    plt.figure(figsize=(10, 6))
    plt.bar(list(avg_acc.keys()), list(avg_acc.values()), color='#16a085', edgecolor='black', alpha=0.8)
    plt.title("Figure: Word Length vs. Accuracy", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Word Length (words)", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("Accuracy (%)", fontsize=14, fontproperties=nyala_prop)
    plt.tight_layout()
    plt.savefig(f"{THESIS_VISUALS_DIR}/word_length_vs_accuracy.png", dpi=300)
    plt.close()
    print(f"✅ Word length vs. accuracy plot saved as '{THESIS_VISUALS_DIR}/word_length_vs_accuracy.png'")

def plot_loss():
    log_file = find_trainer_state(SEARCH_DIR)
    
    if not log_file:
        print("❌ Error: Could not find 'trainer_state.json' anywhere in outputs/.")
        print("   Please check if the 'outputs' folder exists.")
        return

    with open(log_file, 'r') as f:
        data = json.load(f)

    if 'log_history' not in data:
        print("❌ Error: JSON file found but contains no log history.")
        return

    history = data['log_history']
    
    # Extract data points
    train_steps = []
    train_loss = []
    
    for entry in history:
        if 'loss' in entry and 'step' in entry:
            train_steps.append(entry['step'])
            train_loss.append(entry['loss'])

    if not train_steps:
        print("❌ No loss data found in logs.")
        return

    # Plotting
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))

    # Plot the line
    plt.plot(train_steps, train_loss, color='#2980b9', linewidth=2.5, label='Training Loss')

    # Styling with Nyala font
    plt.title("Figure 4.3: Training Loss Convergence (Word-Aware Loss)", fontsize=16, weight='bold', pad=20, fontproperties=nyala_prop)
    plt.xlabel("Training Steps", fontsize=14, fontproperties=nyala_prop)
    plt.ylabel("Cross Entropy Loss", fontsize=14, fontproperties=nyala_prop)
    plt.legend(fontsize=12, prop=nyala_prop)

    # Annotate Start and End
    start_loss = train_loss[0]
    final_loss = train_loss[-1]

    plt.annotate(f'Start: {start_loss:.4f}', xy=(train_steps[0], start_loss), 
                 xytext=(train_steps[0]+500, start_loss+0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontproperties=nyala_prop)
    plt.annotate(f'Final: {final_loss:.4f}', xy=(train_steps[-1], final_loss), 
                 xytext=(train_steps[-1]-3000, final_loss+1.0),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontproperties=nyala_prop)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Loss curve saved as '{OUTPUT_IMAGE}'")
    plt.close()

if __name__ == "__main__":
    plot_loss()
    plot_training_accuracy()
    plot_confusion_matrix()
    plot_wer_vs_char_count()
    plot_cer_distribution()
    plot_word_length_vs_accuracy()