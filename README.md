# NCAIR-DSA: Multi-Lingual Patient Intake Assistant

Voice-powered intake tool for primary healthcare centers. A patient speaks symptoms
in Yoruba, Igbo, or Hausa; the system detects the language, transcribes it (NCAIR
ASR), then structures it into a **full, descriptive English clinical note** plus
**conservative, hedged possible-cause suggestions** (N-ATLaS LLM) — removing the
translation barrier for doctors who don't speak the patient's language. All visits
are linked to patient IDs for traceable clinical history, with exports for both the
full database and individual transcripts.

## Main Goal

Remove language barriers in healthcare by generating full structured English
clinical notes from patient speech in indigenous languages. Doctors see a complete,
editable clinical note — not a translation, but a proper structured clinical
document — with an AI-suggested (clearly hedged) list of possible considerations
alongside it, never presented as a diagnosis.

## Pipeline

1. **Capture** — nurse records patient audio via Gradio (web/mobile friendly)
2. **Detect Language** — audio-based language detection (Whisper's language-ID,
   run on raw audio *before* ASR) suggests Hausa/Igbo/Yoruba; nurse confirms or
   overrides manually
3. **Transcribe** — matching NCAIR ASR model converts speech to text (raw
   transcript is kept and displayed separately, in case of mistranslation)
4. **Structure** — N-ATLaS LLM generates:
   - A full English clinical note (chief complaint, duration, severity, history)
     — grounded strictly in what the patient said, no invented details
   - A separate "Possible Recommendations" field — conservative, hedged
     considerations for the doctor, explicitly not a diagnosis
5. **Extract Keywords** — pulls quick-reference terms from the **English
   structured note** (not the raw non-English transcript) for fast scanning
6. **Doctor Review** — tabbed review screen: Clinical Note (primary) → Possible
   Recommendations (secondary) → Keywords (minor reference); doctor edits and
   saves
7. **Patient Tracking** — saved visit is linked to the patient's ID in the
   database, building traceable history across visits
8. **Export** — full database as CSV, or an individual patient's translated
   note + original transcript as plain text

## Role-Based Interface

The app has two views, toggled by a role selector at the top:

- **Nurse view** — audio capture, patient ID, language detection/confirmation only.
  No access to clinical note editing, recommendations, or exports.
- **Doctor view** — full tabbed review, save, patient history lookup, CSV/TXT
  exports.

This mirrors how the tool would realistically be used in a clinic: intake and
clinical review are different jobs and shouldn't share a screen.

## Repo Structure

```text
ncair-dsa/
├── app.py                        # Gradio entry point — roles, pipeline, exports
├── asr/
│   ├── transcribe.py              # ASR: preprocess -> denoise -> amplify -> transcribe
│   └── README.md                  # ASR
Keyword extraction (runs on English note)
│   ├── audio_language_detect.py   # ASR
module documentation
├── nlp/
│   ├── structure_note.py          # N-ATLaS: clinical note + possible recommendations
│   ├── extract_keywords.py        # Audio-based language ID, runs BEFORE ASR
│   ├── STRUCTURING_GUIDE.md       # LLM structuring module documentation
│   └── QUICKREF.md                # Quick reference for the structuring module
├── templates/
│   └── clinical_note.py           # Patient/visit schema, save/retrieve, CSV export
├── data/
│   └── notes.db                   # SQLite (persisted to Google Drive when mounted)
├── tests/
│   └── test_pipeline.py           # Unit tests: keyword extraction, JSON parsing, DB
├── requirements.txt
├── .env.example
└── README.md
