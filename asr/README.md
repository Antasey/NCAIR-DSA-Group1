# ASR Module — NCAIR-DSA

## What This Does

Converts patient audio (spoken in Hausa, Igbo, or Yoruba) into raw text using NCAIR's Whisper-Small fine-tuned models. This raw transcript is then passed to the LLM structuring module (`nlp/structure_note.py`), which translates and structures it into an English clinical note.

**This module's job stops at producing a clean, accurate transcript.** Translation and structuring happen downstream.

---

## Models Used

| Language | Model ID |
|---|---|
| Hausa | `NCAIR1/Hausa-ASR` |
| Igbo | `NCAIR1/Igbo-ASR` |
| Yoruba | `NCAIR1/Yoruba-ASR` |

All are Whisper-Small fine-tunes released by Awarri Technologies / NCAIR, and expect **16kHz mono audio** as input.

---

## Processing Pipeline

Raw audio rarely comes in ready-to-transcribe, especially recorded in a hospital waiting room. The full pipeline runs several cleanup steps before the audio ever reaches the ASR model:

```
Raw Audio Input
      │
      ▼
1. Preprocessing (resample to 16kHz mono)
      │
      ▼
2. Silence Detection (catch failed/empty recordings early)
      │
      ▼
3. Noise Reduction (strip background hospital noise)
      │
      ▼
4. Volume Amplification/Normalization (boost quiet speech)
      │
      ▼
5. ASR Transcription (NCAIR Whisper-Small model, per language)
      │
      ▼
Raw Transcript (text, in the spoken language)
```

### Step 1: Preprocessing — Resample to 16kHz Mono

Whisper-based models expect 16kHz mono audio. If the recording comes in at a different sample rate or in stereo, transcription accuracy drops even though nothing errors out. This step normalizes every input to the expected format before anything else happens.

### Step 2: Silence Detection

Checks the audio's signal level (RMS) before wasting a transcription call on an empty or near-silent recording — for example if the patient hadn't started speaking yet, or the mic failed to capture. If audio is too quiet, an error is raised immediately with a clear message, instead of the model returning empty or hallucinated text.

### Step 3: Noise Reduction

Estimates the background noise profile (waiting-room chatter, ambient hum, etc.) and subtracts it from the recording. This is cleanup applied to a full recorded clip — not real-time noise cancellation — and works best when there's a consistent, learnable noise pattern in the background.

**Known limitation:** this helps, but doesn't fully solve noisy-environment transcription. The N-ATLaS/NCAIR model card already discloses degraded performance in noisy environments as a known weakness. This step mitigates it; it doesn't eliminate it.

### Step 4: Volume Amplification / Normalization

Brings quiet patient speech up to a consistent volume level without distorting or clipping louder background sounds. Useful when a patient speaks softly relative to ambient noise.

### Step 5: Transcription

The cleaned audio is passed to the matching NCAIR Whisper-Small model for the selected language. Long recordings are chunked (`chunk_length_s=30`) so accuracy doesn't degrade on longer patient statements.

---

## Error Handling

Unlike the original bare-bones version, this pipeline **fails loudly, not silently**:

- Silent/empty audio → raises `ValueError` with a clear message before transcription is even attempted
- Any step in the pipeline failing → logged with `logger.error()` and re-raised, so the calling code (`app.py`) can catch it and show the doctor a real error message instead of a blank note

This matters because a silent failure (empty string returned, no error) looks identical to "the patient didn't say anything," which is a much harder bug to diagnose than a clear exception.

---

## Accuracy Testing (Word Error Rate)

Before trusting this pipeline for a demo or real use, its output should be tested against known-correct reference transcripts using Word Error Rate (WER):

```python
from jiwer import wer

hypothesis = transcribe_audio(audio_path, language)
error_rate = wer(reference_text, hypothesis)
```

- `error_rate = 0.0` → perfect match
- Above roughly `0.3`–`0.4` → worth investigating (bad audio quality, wrong language selected, or a genuine model limitation)

Testing should be done per language, since accuracy is not uniform — Hausa currently performs strongest, Yoruba weakest, per N-ATLaS's published human evaluation scores.

---

## Usage

```python
from asr.transcribe import transcribe_audio

transcript = transcribe_audio("patient_audio.wav", "Hausa")
print(transcript)  # raw text, in Hausa — NOT translated yet
```

**Important:** the output of this function is still in the original spoken language. Translation to English happens in the next stage (`nlp/structure_note.py`), not here. Do not run keyword extraction or English-only text processing directly on this output — see the note below.

---

## Known Interaction With Downstream Modules

**Keyword extraction runs on the translated English note, not on this module's raw output.** Since `extract_keywords()` matches against English clinical terms (`pain`, `fever`, `cough`, etc.), running it directly on a Hausa/Igbo/Yoruba transcript from this module will not produce useful matches. The correct flow is:

```
transcribe_audio()  →  raw transcript (local language)
       │
       ▼
structure_note()    →  English structured note
       │
       ▼
extract_keywords()  →  runs on the English note, not the raw transcript
```

This module only owns the first step.

---

## Dependencies

```
transformers
torch
librosa
soundfile
noisereduce
pydub
jiwer
```

Add these to `requirements.txt` if not already present.

---

## Testing Checklist

- [ ] Test with real Hausa, Igbo, and Yoruba audio samples (not just English)
- [ ] Confirm silence detection correctly flags empty recordings
- [ ] Run WER tests against 5–10 known reference transcripts per language
- [ ] Confirm noise reduction improves (not degrades) transcription on a genuinely noisy sample
- [ ] Confirm longer recordings (60+ seconds) still transcribe accurately with chunking enabled
- [ ] Confirm errors surface clearly in `app.py` rather than failing silently

---

## Known Limitations

- Dialectal and accent variation is not fully accounted for — the underlying models have disclosed sensitivity to this
- Performance on children's speech is weaker
- Code-switching (patients mixing English and local language mid-sentence) is not reliably handled
- Noise reduction is post-processing, not real-time — it cannot fully compensate for a very poor original recording
