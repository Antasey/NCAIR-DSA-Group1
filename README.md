# NCAIR-DSA: Multi-Lingual Patient Intake Assistant

Voice-powered intake tool for primary healthcare centers. Patient speaks
symptoms in Yoruba, Igbo, or Hausa; the system transcribes it (NCAIR ASR),
structures it into an English clinical note (N-ATLaS LLM), extracts and
highlights clinical keywords, and stores traceable patient history in the database.

## Pipeline
1. **Capture** — patient records audio via Gradio (web/mobile friendly)
2. **Transcribe** — NCAIR/N-ATLaS ASR model (per-language) converts speech to text
3. **Structure** — N-ATLaS-LLM turns transcript into structured English note (JSON)
4. **Extract Keywords** — pull out clinical entities (symptoms, severity, duration,
   anatomical sites) from the raw transcript
5. **Highlight & Review** — display structured note with keywords highlighted in
   yellow so doctor can spot key details instantly
6. **Patient Tracking** — save visit to patient's clinical history (linked by patient ID)
   in the database for traceable history across multiple visits

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
```bash
git clone <repo-url>
cd ncair-dsa
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # fill in your HF_TOKEN
python app.py                 # launches Gradio web interface
```

## Models
- ASR: `NCAIR1/Hausa-ASR`, `NCAIR1/Igbo-ASR`, `NCAIR1/Yoruba-ASR` (Hugging Face)
- Structuring: `NCAIR1/N-ATLaS` (8B params — needs GPU with 8GB+ VRAM for
  4-bit, or use a hosted HF Inference Endpoint via `NATLAS_ENDPOINT_URL`)

## Key Features
- **Keyword Extraction**: automatically pulls symptoms, severity, duration, and
  anatomical sites from patient speech
- **Visual Highlighting**: keywords highlighted in yellow in the doctor review screen
  for quick scanning
- **Patient Tracking**: each patient has a unique ID; all visits linked to that ID
  for complete traceable clinical history
- **Database Export**: generated patient records are exportable (JSON) for doctor
  to input into their own system

## Team Workflow
- Work on feature branches, not `main` — open a PR to merge
- Keep `requirements.txt` in sync if you add a dependency
- Never commit `.env`, audio files, or model weights (see `.gitignore`)

## Module Ownership
| Module | Folder | Owner(s) |
|---|---|---|
| ASR integration | `asr/` | 3 people — one per language testing |
| LLM structuring | `nlp/structure_note.py` | 2 people — prompt engineering + LLM connection |
| Keyword extraction | `nlp/extract_keywords.py` | 1 person — expand keywords, tune highlighting |
| Frontend/UI (Gradio) | `app.py` | 2 people — audio UI + patient ID form + review screen |
| Patient history/DB | `templates/clinical_note.py`, `data/` | 1 person — patient tracking + visit history |
| Docs/testing | `tests/`, `README.md` | 1 person — keep README updated, test pipeline |

## Pilot Language
Starting with Hausa — strongest current ASR/LLM performance among the
three supported languages.