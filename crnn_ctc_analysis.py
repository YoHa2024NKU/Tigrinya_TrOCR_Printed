"""
crnn_ctc_analysis.py

- Loads training log and test results from CRNN-CTC baseline
- Saves results to a JSON file
- Plots training/validation loss and accuracy curves
- Prints a summary table for publication
"""
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ----------------------
# 1. CONFIGURATION
# ----------------------
LOG_FILE = "crnn_ctc_training_log.json"
RESULTS_FILE = "crnn_ctc_results.json"
PLOT_DIR = "crnn_ctc_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# ----------------------
# 2. LOAD LOGS & RESULTS
# ----------------------
def load_logs():
    if not os.path.exists(LOG_FILE):
        print(f"Log file {LOG_FILE} not found. Please modify your training script to save logs.")
        return None
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log = json.load(f)
    return log

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file {RESULTS_FILE} not found. Please modify your training script to save results.")
        return None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
    return results

# ----------------------
# 3. PLOTTING
# ----------------------
def plot_curves(log):
    epochs = np.arange(1, len(log['train_loss'])+1)
    plt.figure(figsize=(8,5))
    plt.plot(epochs, log['train_loss'], label='Train Loss')
    plt.plot(epochs, log['dev_cer'], label='Dev CER')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / CER')
    plt.title('CRNN-CTC Training Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'crnn_ctc_training_curve.png'))
    plt.close()

    if 'dev_acc' in log:
        plt.figure(figsize=(8,5))
        plt.plot(epochs, log['dev_acc'], label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('CRNN-CTC Dev Accuracy Curve')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, 'crnn_ctc_dev_accuracy.png'))
        plt.close()

# ----------------------
# 4. SUMMARY TABLE
# ----------------------
def print_summary_table(results):
    print("\nCRNN-CTC Baseline Results (Test Set):")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| CER    | {results['test_cer']:.4f} |")
    print(f"| WER    | {results['test_wer']:.4f} |")
    print(f"| Acc    | {results['test_acc']:.4f} |")

# ----------------------
# 5. MAIN
# ----------------------
def main():
    log = load_logs()
    results = load_results()
    if log:
        plot_curves(log)
    if results:
        print_summary_table(results)
        # Save summary table to markdown
        with open(os.path.join(PLOT_DIR, 'crnn_ctc_summary.md'), 'w', encoding='utf-8') as f:
            f.write("| Metric | Value |\n|--------|-------|\n")
            f.write(f"| CER    | {results['test_cer']:.4f} |\n")
            f.write(f"| WER    | {results['test_wer']:.4f} |\n")
            f.write(f"| Acc    | {results['test_acc']:.4f} |\n")

if __name__ == "__main__":
    main()
