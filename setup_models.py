"""
Setup script — download and cache the N-ATLaS model BEFORE running the app.

Run this once before starting the Gradio app:
    python setup_models.py

If the model is gated/private, set your Hugging Face token first:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    python setup_models.py

Or in Colab:
    import os
    os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxx"
    !python setup_models.py
"""

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi

MODEL_DIR = Path("models/N-ATLaS")
MODEL_ID = "NCAIR1/N-ATLaS"
HF_TOKEN = os.getenv("HF_TOKEN")


def check_access():
    """Verify we can access the repo before trying to download."""
    api = HfApi(token=HF_TOKEN)
    try:
        api.model_info(MODEL_ID)
        print(f"✓ Access confirmed for {MODEL_ID}")
        return True
    except Exception as e:
        print(f"✗ Cannot access {MODEL_ID}: {e}")
        if "401" in str(e):
            print("\n🔑 This model requires a Hugging Face token.")
            print("   Get one at: https://huggingface.co/settings/tokens")
            print("   Then run:   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx")
            print("   Or in Colab: os.environ['HF_TOKEN'] = 'hf_...'")
        return False


def download_model():
    """Download N-ATLaS from Hugging Face to the local models/ folder."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not check_access():
        sys.exit(1)

    print(f"\n📥 Downloading {MODEL_ID} to {MODEL_DIR} ...")
    print("   This is ~5GB and may take 5–10 minutes.\n")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
    )
    print(f"\n✅ Model ready at {MODEL_DIR}")
    print("   You can now run: python app.py")


if __name__ == "__main__":
    download_model()
