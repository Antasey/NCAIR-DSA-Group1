# NCAIR-DSA: Multi-Lingual Patient Intake Assistant

Voice-powered intake tool for primary healthcare centers. Patient speaks symptoms 
in Yoruba, Igbo, or Hausa; the system transcribes it (NCAIR ASR), then structures 
it into a **full, complete English clinical note** (N-ATLaS LLM) that removes the 
translation barrier entirely. Keywords are extracted as a supporting **quick-reference 
sidebar** to help doctors scan key clinical points during review. All visits are 
linked to patient IDs for traceable clinical history.

## Main Goal
Remove language barriers in healthcare by generating full structured English clinical 
notes from patient speech in indigenous languages. Doctors see a complete, editable 
clinical note — not a translation, but a proper structured clinical document.

## Pipeline
1. **Capture** — patient records audio via Gradio (web/mobile friendly)
2. **Transcribe** — NCAIR/N-ATLaS ASR model (per-language) converts speech to text
3. **Structure** — N-ATLaS-LLM generates full English clinical note (chief complaint, 
   duration, severity, history) — THIS IS THE MAIN DELIVERABLE
4. **Extract Keywords** — pull out clinical entities (symptoms, severity, duration, 
   anatomical sites) from the raw transcript as supporting reference only
5. **Review** — doctor reviews and edits the full clinical note; keywords appear 
   in a read-only sidebar for quick scanning
6. **Save** — edited note is saved to patient's clinical history (linked by patient ID) 
   for complete traceable record across multiple visits

## Repo Structure
```
ncair-dsa/
├── app.py                    # Gradio entry point: audio capture, transcribe,
│                             # keyword highlight, patient record save
├── asr/
│   └── transcribe.py         # loads ASR model per language, returns text
├── nlp/
│   ├── structure_note.py     # prompts N-ATLaS-LLM, parses JSON output
│   └── extract_keywords.py   # extract + highlight clinical keywords
├── templates/
│   └── clinical_note.py      # patient schema, clinical history tracking
├── data/
│   └── notes.db              # SQLite patient records (not committed)
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── .env.example
└── README.md
```

## Setup
LOCALLY
```bash
git clone <repo-url>
cd ncair-dsa
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in your HF_TOKEN
python app.py                 # launches Gradio web interface
```
## 🚀 Running on Google Colab (Recommended for Low-Spec PCs)

If your local computer does not have a powerful GPU or sufficient RAM to load the model locally, you can run this entire project completely in the cloud using Google Colab's free T4 GPU.

### 1. Set Up Your Cloud GPU
1. Open [Google Colab](https://google.com) and create a **New Notebook**.
2. Go to the top menu: **Runtime** ➔ **Change runtime type**.
3. Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

### 2. Add Your Hugging Face Token to Colab
To securely use your Hugging Face credentials without exposing them in your code:
1. Click the **Key icon** (Secrets) on the left sidebar of Google Colab.
2. Click **Add new secret**.
3. Set the **Name** to: `HF_TOKEN`
4. Set the **Value** to: *Your actual Hugging Face Access Token (`hf_...`)*
5. Switch on the toggle for **Notebook access**.

### 3. Clone and Install Dependencies
Paste and run the following block of code in your first Colab cell:

```python
# Clone the repository
!git clone <YOUR-REPO-URL>

# Move into the project directory
%cd ncair-dsa

# Install all required dependencies
!pip install -r requirements.txt
```

### 4. Configure Environment & Launch
Create a new code cell, paste the script below, and run it to launch the Gradio web interface via a cloud endpoint:

```python
import os
from google.colab import userdata

# Load your token safely from Colab Secrets
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')

# Use Hugging Face's hosted cloud architecture for inference
os.environ["NATLAS_ENDPOINT_URL"] = "https://huggingface.co"

# Launch the interface with a public shareable URL
!python app.py --share
```

### 5. Access the Interface
Once the script finishes executing, look at the terminal output for a URL ending in **`.gradio.live`**. Click that public link to open and use your application in a new browser tab.

## Info
Create your own branch from main and push commits only to your branch, Deborah will have to approve to merge when you put in a pull request for ease of access and proper coordination. 
## Models
- ASR: `NCAIR1/Hausa-ASR`, `NCAIR1/Igbo-ASR`, `NCAIR1/Yoruba-ASR` (Hugging Face)
- Structuring: `NCAIR1/N-ATLaS` (8B params — needs GPU with 8GB+ VRAM for
  4-bit, or use a hosted HF Inference Endpoint via `NATLAS_ENDPOINT_URL`)

## Key Features
- **Full Clinical Notes**: converts unstructured patient speech into complete, 
  structured English clinical notes with chief complaint, duration, severity, 
  and relevant history — removes the translation barrier entirely
- **Editable Doctor Review**: doctor can edit any part of the generated note before 
  saving; full control over what goes into the patient record
- **Quick-Reference Keywords** (supporting): extracted clinical keywords appear as 
  a non-editable sidebar for doctors to quickly scan key clinical points without 
  re-reading the full note
- **Patient Tracking**: each patient has a unique ID; all visits linked to that ID 
  for complete traceable clinical history across multiple intake sessions
- **Multi-Language Support**: Hausa, Igbo, Yoruba — pilot starting with Hausa

## Team Workflow
- Work on feature branches, not `main` — open a PR to merge
- Keep `requirements.txt` in sync if you add a dependency
- Never commit `.env`, audio files, or model weights (see `.gitignore`)

## Module Ownership
| Module | Folder | Owner(s) | Priority |
|---|---|---|---|
| **LLM Structuring (CORE)** | `nlp/structure_note.py` | 2-3 people | 🔴 HIGHEST — generates the main clinical note |
| ASR integration | `asr/` | 2-3 people | 🔴 HIGH — speech input pipeline |
| Frontend/UI (Gradio) | `app.py` | 2 people | 🟡 MEDIUM — user interface |
| Patient History/DB | `templates/clinical_note.py`, `data/` | 1 person | 🟡 MEDIUM — persistence layer |
| **Keyword Extraction (SUPPORTING)** | `nlp/extract_keywords.py` | 1 person | 🟢 LOW — enhances review, not core |
| Docs/Testing | `tests/`, `README.md` | 1 person | 🟢 LOW — integration + documentation |

## Pilot Language
Starting with Hausa — strongest current ASR/LLM performance among the
three supported languages.
