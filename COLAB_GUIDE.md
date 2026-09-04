# NCAIR-DSA — Google Colab Setup Guide

## ✅ Yes, this runs perfectly on Google Colab

All model downloads use **Colab's internet & GPU**, not yours.

---

## 🔑 IMPORTANT: Hugging Face Token Required

The `NCAIR1/N-ATLaS` model requires a Hugging Face token to download.

1. Get a free token at: https://huggingface.co/settings/tokens
2. Copy the token (starts with `hf_`)

---

## 📓 Run these cells in order

### Cell 1 — Set your Hugging Face token
```python
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxxxxxxxxx"  # <-- paste your token here
```

### Cell 2 — Mount Google Drive (optional, for data persistence)
```python
from google.colab import drive
drive.mount('/content/drive')
```
> This keeps your `notes.db` and downloaded model between sessions.
> Skip if you don't need persistence.

---

### Cell 3 — Clone & install
```python
!git clone https://github.com/Antasey/NCAIR-DSA-Group1.git
%cd NCAIR-DSA-Group1
!pip install -q -r requirements.txt
```

---

### Cell 4 — Download N-ATLaS model (~5GB, uses Colab's network)
```python
%cd /content/NCAIR-DSA-Group1
!python setup_models.py
```
> This downloads once. If you mounted Drive, the model caches there too.
> Takes 3–5 minutes on Colab.

---

### Cell 5 — Launch the app
```python
%cd /content/NCAIR-DSA-Group1
!python app_colab.py
```
> Look for the **public URL**: `https://xxxx.gradio.live`
> Open that in your browser — not the `127.0.0.1` link.

---

## 💾 Optional: Persist model on Drive

If you don't want to re-download the model every session:

```python
# Run once after setup_models.py finishes
import shutil
shutil.copytree("models/N-ATLaS", "/content/drive/MyDrive/NCAIR-DSA/models/N-ATLaS")
```

Then in future sessions, skip Cell 4 and symlink instead:
```python
!mkdir -p models
!ln -s /content/drive/MyDrive/NCAIR-DSA/models/N-ATLaS models/N-ATLaS
```

---

## 🖥️ GPU Check

Colab free gives a T4 GPU (16GB VRAM). The 4-bit quantized N-ATLaS uses ~6GB.

```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

---

## ⚠️ Known Colab Quirks

| Issue | Fix |
|---|---|
| Audio recording fails | Use **Upload** instead of Microphone (browser permission limits) |
| Session times out after ~90 min | Keep the Colab tab active; use Drive mount for data |
| `models/N-ATLaS` not found | Re-run Cell 4; check Drive symlink if using persistence |
| Gradio link shows `ERR_CONNECTION_REFUSED` | You opened the local URL — use the `https://xxxx.gradio.live` link |
| **401 Unauthorized from Hugging Face** | Your `HF_TOKEN` is missing or invalid — check Cell 1 |

---

## 📊 What uses YOUR network vs Colab's

| Action | Uses YOUR network? | Uses Colab's? |
|---|---|---|
| Opening the Gradio link | ✅ Yes (your browser) | ❌ No |
| Downloading model (setup_models.py) | ❌ No | ✅ Yes |
| Transcribing audio | ❌ No | ✅ Yes (GPU) |
| Generating clinical note | ❌ No | ✅ Yes (GPU) |
| Saving to database | ❌ No | ✅ Yes |
| Exporting CSV | ❌ No | ✅ Yes |
