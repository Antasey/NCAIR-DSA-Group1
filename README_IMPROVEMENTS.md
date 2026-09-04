# NCAIR-DSA Improved Codebase

## What's New

### 1. Transformers-based N-ATLaS (not llama-cpp)
- Model loaded via `transformers` + `bitsandbytes` 4-bit quantization
- **Pre-download required:** Run `python setup_models.py` before starting the app
- Model downloads to `models/N-ATLaS/` and is cached there

### 2. Nurse Sees English Translation
- After processing, the nurse intake page shows:
  - English clinical note preview (Chief Complaint, Duration, Severity, History)
  - Original transcript for reference
- No more "black box" — the nurse verifies the translation before it goes to the doctor queue

### 3. No Empty LLM Responses
- `generate_text()` raises `RuntimeError` if the model returns empty text
- `structure_note()` retries up to 2 times on failure
- `app.py` shows a clear error message to the user instead of blank fields
- Keyword extraction falls back gracefully but never crashes the app

### 4. Setup Script
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download N-ATLaS model (one-time, ~5GB)
python setup_models.py

# 3. Run the app
python app.py
```

### 5. File Changes
| File | Change |
|---|---|
| `setup_models.py` | **NEW** — downloads N-ATLaS to `models/N-ATLaS/` |
| `app.py` | Nurse preview panel, no empty LLM guard, model pre-check |
| `nlp/structure_note.py` | Transformers-based, retry logic, empty-field guard |
| `nlp/extract_keywords.py` | Uses `generate_text()` from structure_note, dedup |
| `asr/transcribe.py` | Temp dir isolation, GPU mem check |
| `nlp/audio_language_detect.py` | Tiny model, thread-safe, never returns None |
| `templates/clinical_note.py` | Transactions, indexes, context manager |
| `tests/` | Mocked LLM, 12 DB tests, fixed imports |
| `requirements.txt` | Added `accelerate`, `bitsandbytes` |
| `.gitignore` | Added Python, model, and data ignores |

## Quick Start
```bash
pip install -r requirements.txt
python setup_models.py
python app.py
```
