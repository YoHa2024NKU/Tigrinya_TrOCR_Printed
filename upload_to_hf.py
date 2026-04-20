import os
from huggingface_hub import HfApi




MY_TOKEN = ""  # Paste  token here
REPO_ID = "Yonatanhaile2026/tigrinya-trocrprinted"  #  Username / Model Name
LOCAL_FOLDER = "outputs/fast_model_printed/best_model" 
# -------------------------

api = HfApi()

print(f"Starting upload to {REPO_ID}...")

# This command uploads everything in the folder and creates the repo if missing
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=MY_TOKEN,
    ignore_patterns=["checkpoint-*"]  
)

print("Upload Completed! Check your private repo.")