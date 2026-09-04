# MediVoice — Multi-Lingual Patient Intake Assistant

**NCAIR-DSA Group 1 Project**

A voice-powered clinical intake system that enables patients to describe symptoms in **Yoruba, Igbo, or Hausa**, and automatically structures the conversation into a standardized **English clinical note** for doctor review.

Built during the **NCAIR & NITDA Data Science and AI Residency (DSA) Programme**.

---

## 📋 Table of Contents

- [What the App Does](#what-the-app-does)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [Team Members](#team-members)
- [Acknowledgements](#acknowledgements)

---

## What the App Does

MediVoice bridges the language gap between patients and healthcare providers by combining **Automatic Speech Recognition (ASR)**, **Large Language Models (LLM)**, and a **clinical review workflow**.

### Core Functionality

| Feature | Description |
|---|---|
| **🎙️ Voice Intake** | Patients record symptoms in their native language (Hausa, Igbo, or Yoruba) via microphone or audio upload. |
| **🔍 Language Detection** | Automatically detects the spoken language from audio before transcription. Flags code-switching when detected. |
| **📝 ASR Transcription** | Uses NCAIR fine-tuned Whisper-Small models to transcribe audio into text while preserving the original language. |
| **🤖 Clinical Note Structuring** | Feeds the transcript into the **N-ATLaS** LLM to generate a structured English clinical note with 5 fields: Chief Complaint, Duration, Severity, History, and Possible Recommendations. |
| **🔑 Keyword Extraction** | Automatically extracts clinical keywords (symptoms, duration, severity, anatomical sites) from the English note for quick doctor scanning. |
| **👩‍⚕️ Nurse Preview** | Nurses see the English translation immediately after processing, with the original transcript preserved for verification. |
| **🩺 Doctor Review Queue** | Doctors access a queue of pending visits, review structured notes, edit fields, and confirm or override AI-generated recommendations. |
| **📤 CSV Export** | Individual visits can be exported as CSV files for record-keeping or EHR integration. |
| **📊 Dashboard** | Real-time statistics showing total patients, clinical visits, and pending reviews. |

### Workflow

```
Patient speaks (Yoruba/Igbo/Hausa)
        ↓
Language Detection (Whisper tiny)
        ↓
ASR Transcription (NCAIR Whisper-Small)
        ↓
LLM Structuring (N-ATLaS → English clinical note)
        ↓
Keyword Extraction (N-ATLaS)
        ↓
Nurse Preview (verify English translation)
        ↓
Save to Database (SQLite)
        ↓
Doctor Review Queue
        ↓
Doctor Edits & Finalizes
        ↓
Export / Archive
```

### Safety & Ethics

- **No diagnosis**: The LLM only provides "possible recommendations" — clearly labeled as non-diagnostic.
- **Grounded in patient speech**: Fields 1–4 (Chief Complaint, Duration, Severity, History) are strictly derived from what the patient said. No hallucination.
- **Human-in-the-loop**: Every note is reviewed by a doctor before finalization.
- **Data privacy**: All data is stored locally in SQLite — no cloud dependency for patient records.

---

## System Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Gradio UI     │────▶│   app.py        │────▶│   SQLite DB     │
│  (Dashboard,    │     │  (Orchestrator) │     │  (notes.db)     │
│   Intake,       │     └─────────────────┘     └─────────────────┘
│   Review)       │              │
└─────────────────┘              ▼
                        ┌─────────────────┐
                        │   asr/          │
                        │  transcribe.py  │◄── NCAIR Whisper-Small
                        └─────────────────┘
                        ┌─────────────────┐
                        │   nlp/          │
                        │  audio_language │◄── Whisper tiny
                        │  _detect.py     │
                        ├─────────────────┤
                        │  structure_note │◄── N-ATLaS LLM
                        │  .py            │
                        ├─────────────────┤
                        │  extract_       │◄── N-ATLaS LLM
                        │  keywords.py    │
                        └─────────────────┘
                        ┌─────────────────┐
                        │   templates/    │
                        │  clinical_note  │◄── Patient schema,
                        │  .py            │    visit tracking,
                        │                 │    review queue
                        └─────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **UI Framework** | Gradio 4.x |
| **ASR Models** | NCAIR Whisper-Small fine-tunes (Hausa, Igbo, Yoruba) |
| **Language Detection** | OpenAI Whisper (tiny) |
| **LLM** | N-ATLaS (loaded via Transformers + 4-bit quantization) |
| **Audio Preprocessing** | librosa, noisereduce, pydub |
| **Database** | SQLite3 |
| **Testing** | pytest |
| **Environment** | Python 3.10+ |

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Hugging Face account (for gated model access)

### 1. Clone the repository

```bash
git clone https://github.com/Antasey/NCAIR-DSA-Group1.git
cd NCAIR-DSA-Group1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Hugging Face token

The N-ATLaS model requires authentication.

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

Get your token at: https://huggingface.co/settings/tokens

### 4. Download the N-ATLaS model

```bash
python setup_models.py
```

This downloads ~5GB of model weights to `models/N-ATLaS/`. It only needs to run once.

---

## Running the App

### Local Machine

```bash
python app.py
```

The app will launch with a public Gradio link (`https://xxxx.gradio.live`) that you can share.

### Google Colab

```python
# In a Colab cell
%cd /content/NCAIR-DSA-Group1
!python app_colab.py
```

> **Note**: On Colab, use the **Upload** tab for audio instead of the microphone (browser permission limitations).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- LLM output parsing and JSON validation
- Keyword extraction (with mocked LLM)
- Database CRUD operations
- Visit queue ordering and review workflow

---

## Team Members — Group 1

| Name | Role |
|---|---|
| **Confidence Martins Aweh** | Team Member |
| **Deborah Tony-Owakah** | Group Leader |
| **Favour Oseghale** | Team Member |
| **Jesutomi Santa** | Team Member |
| **Oluwafikolami Ejidare** | Team Member |
| **Osinachi Ezimah** | Team Member |
| **Samuel Omoleye** | Team Member |

*All team members contributed to the design, development, and testing of this application.*

---

## Acknowledgements

We extend our sincere gratitude to:

- **Mr. Victor Rizama** — for his guidance, mentorship, and technical oversight throughout the residency programme.
- **Mr. Stephen Ayuba** — for his invaluable support, feedback, and encouragement during the development of this project.
- **NCAIR (National Centre for Artificial Intelligence and Robotics)** — for providing the computational resources, datasets, and AI infrastructure that made this project possible.
- **NITDA (National Information Technology Development Agency)** — for facilitating the Data Science and AI Residency Programme and creating an enabling environment for innovation and capacity building in Nigeria.

---

## License

This project was developed for educational purposes during the NCAIR-NITDA DSA Residency Programme.

---

*Built with ❤️ by NCAIR-DSA Group 1.*
