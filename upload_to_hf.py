from huggingface_hub import HfApi

# --- YOUR DETAILS HERE ---
MY_TOKEN = "     "  # Paste your token here
REPO_ID = "Yonatanhaile2026/tigrinya-trocr-model"  # Your Username / Model Name
LOCAL_FOLDER = "outputs/fast_model" # The folder on your PC where the heavy files are
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

print("Upload Complete! Check your private repo.")