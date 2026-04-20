import os
from huggingface_hub import HfApi



# --- YOUR DETAILS HERE ---
MY_TOKEN = "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"  # Paste your token here
REPO_ID = "Yonatanhaile2026/tigrinya-trocrprinted"  # Your Username / Model Name
LOCAL_FOLDER = "outputs/fast_model_printed/best_model" # The folder on your PC where the heavy files are
# -------------------------

api = HfApi()

print(f"Starting upload to {REPO_ID}...")

# This command uploads everything in the folder and creates the repo if missing
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=MY_TOKEN,
    ignore_patterns=["checkpoint-*"]  # <--- ADD THIS LINE
)

print("Upload Completed! Check your private repo.")