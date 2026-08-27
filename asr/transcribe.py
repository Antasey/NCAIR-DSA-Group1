"""
ASR Stage — converts patient audio (Hausa/Igbo/Yoruba) into raw text using
NCAIR's Whisper-Small fine-tuned models.

Pipeline: preprocess -> silence check -> noise reduction -> amplification ->
transcription. Fails loudly (raises exceptions), not silently.

This module's job stops at producing a clean, accurate transcript. Translation
and structuring into an English clinical note happens downstream in
nlp/structure_note.py — NOT here. Keyword extraction should also run on the
translated English note, not on this module's raw (non-English) output.

See asr/README.md for full documentation of the processing pipeline.
"""

import logging
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

_asr_pipelines = {}


# ============================================================================
# PREPROCESSING
# ============================================================================

def preprocess_audio(audio_path: str, output_path: str = "preprocessed.wav"):
    """Resample to 16kHz mono — the format Whisper-based ASR models expect."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sf.write(output_path, audio, sr)
    return output_path, audio, sr


def is_audio_too_quiet(audio, silence_threshold: float = 0.01) -> bool:
    """Check if audio is essentially silent (likely a failed recording)."""
    rms = np.sqrt(np.mean(audio**2))
    return rms < silence_threshold


def reduce_noise(audio, sr):
    """Reduce background noise using spectral gating. Helps with hospital/
    waiting-room background chatter and ambient hum. This is cleanup applied
    to the full recorded clip, not real-time noise cancellation."""
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)


def amplify_audio(audio_path: str, output_path: str = "amplified.wav") -> str:
    """Normalize volume — brings quiet patient speech up without distorting
    louder background sounds."""
    audio = AudioSegment.from_file(audio_path)
    normalized = normalize(audio)
    normalized.export(output_path, format="wav")
    return output_path


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_asr_pipeline(language: str):
    """Load (or reuse) the ASR pipeline for the given language."""
    if language not in _asr_pipelines:
        model_id = MODEL_MAP[language]
        logger.info(f"Loading ASR model for {language}: {model_id}")
        _asr_pipelines[language] = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            chunk_length_s=30,  # handles longer recordings without truncating/degrading
        )
    return _asr_pipelines[language]


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def transcribe_audio(audio_path: str, language: str) -> str:
    """
    Full pipeline: preprocess -> silence check -> denoise -> amplify -> transcribe.

    Args:
        audio_path: Path to the patient's audio file
        language: One of "Hausa", "Igbo", "Yoruba"

    Returns:
        Raw transcript text, in the spoken language (NOT translated to English yet)

    Raises:
        ValueError: if the audio is silent/empty
        Exception: any other failure in the pipeline (logged before re-raising)
    """
    try:
        # Step 1: preprocess (resample to 16kHz mono)
        pre_path, audio, sr = preprocess_audio(audio_path)

        # Step 2: check for silence/empty recording — fail fast, don't waste
        # a transcription call on dead air
        if is_audio_too_quiet(audio):
            raise ValueError("No audio detected — recording may have failed or mic was silent")

        # Step 3: noise reduction
        cleaned = reduce_noise(audio, sr)
        sf.write("temp_cleaned.wav", cleaned, sr)

        # Step 4: amplify/normalize
        final_path = amplify_audio("temp_cleaned.wav", "temp_final.wav")

        # Step 5: transcribe
        asr = get_asr_pipeline(language)
        result = asr(final_path)
        text = result["text"]

        if not text or not text.strip():
            logger.warning("ASR returned empty text despite audio passing silence check")

        return text

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
