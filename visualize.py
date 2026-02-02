import json
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

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
    
    # Styling
    plt.title("Figure 4.3: Training Loss Convergence (Word-Aware Loss)", fontsize=16, weight='bold', pad=20)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Cross Entropy Loss", fontsize=14)
    plt.legend(fontsize=12)
    
    # Annotate Start and End
    start_loss = train_loss[0]
    final_loss = train_loss[-1]
    
    plt.annotate(f'Start: {start_loss:.4f}', xy=(train_steps[0], start_loss), 
                 xytext=(train_steps[0]+500, start_loss+0.5),
                 arrowprops=dict(facecolor='red', shrink=0.05))
                 
    plt.annotate(f'Final: {final_loss:.4f}', xy=(train_steps[-1], final_loss), 
                 xytext=(train_steps[-1]-3000, final_loss+1.0),
                 arrowprops=dict(facecolor='green', shrink=0.05))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"✅ Loss curve saved as '{OUTPUT_IMAGE}'")
    plt.show()

if __name__ == "__main__":
    plot_loss()