import os
from huggingface_hub import login, snapshot_download

# 1. Login
try:
    login()
    print("✅ Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Login failed: {e}")

# Configuration
DATASET_NAME = "vcharraut/V-Max_Mini_Datasets"  # Corrected ID found via grep
LOCAL_DIR = "./datasets"

def download_dataset():
    print(f"\n🚀 Starting download for: {DATASET_NAME}")
    print(f"📂 Saving to: {os.path.abspath(LOCAL_DIR)}")
    
    try:
        path = snapshot_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            resume_download=True,
            max_workers=8
        )
        print(f"\n✅ Download complete! Data located at: {path}")
    except Exception as e:
        print(f"\n❌ Error downloading {DATASET_NAME}: {e}")

if __name__ == "__main__":
    download_dataset()