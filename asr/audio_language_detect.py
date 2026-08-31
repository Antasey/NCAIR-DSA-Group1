"""
Audio-Based Language Detection — runs BEFORE ASR, on raw audio.

This replaces the earlier transcript-based approach entirely. Instead of
guessing the spoken language(s) from already-transcribed text (which
requires ASR to have already run, and can't help decide which ASR model
to use in the first place), this listens to the raw audio directly using
Whisper's built-in language-ID head, and returns a language decision
BEFORE any NCAIR ASR model is invoked.

Why this matters for the pipeline: currently a nurse manually picks
Hausa/Igbo/Yoruba from a dropdown. This module allows that to become an
auto-detected suggestion instead — with manual override always available,
since detection confidence varies by language (see IMPORTANT NOTE below).

Uses the small/base OpenAI Whisper model (not the NCAIR fine-tunes) purely
for its language-ID head — this is a separate, lightweight model, loaded
only once, distinct from the NCAIR ASR models used for actual transcription.

*** IMPORTANT — VERIFY BEFORE RELYING ON THIS IN THE DEMO ***
Whisper officially supports Hausa, Igbo, and Yoruba for language identification. 
However, detection is not perfectly reliable, especially with accented,
short, or noisy recordings. For this reason, the application uses a confidence
threshold and falls back to manual language selection whenever the detected 
language is unsupported or the confidence is too low. This prevents 
the wrong ASR model from being selected. 
"""

import logging
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maps Whisper's language codes to our three supported NCAIR languages.
# 'ha' = Hausa, 'yo' = Yoruba. Igbo ('ig') is NOT a standard Whisper
# language code — if this mapping doesn't fire for Igbo audio, that
# confirms the gap and manual selection is the correct fallback, not a bug.
WHISPER_CODE_TO_LANGUAGE = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",  # kept here in case a newer Whisper build supports it —
                    # verify empirically, do not assume this works
    "en": "English",
}

_detection_model = None


def _load_detection_model(model_size: str = "base"):
    """Lazy-load the small Whisper model used only for language ID."""
    global _detection_model
    if _detection_model is None:
        logger.info(f"Loading Whisper '{model_size}' model for language detection...")
        _detection_model = whisper.load_model(model_size)
    return _detection_model


def detect_audio_language(audio_path: str, confidence_threshold: float = 0.5) -> dict:
    """
    Detect the spoken language directly from audio, before ASR runs.

    Args:
        audio_path: Path to the patient's audio file
        confidence_threshold: Minimum confidence to trust the auto-detection.
            Below this, we don't guess — we flag for manual selection instead.
            This threshold matters most for languages Whisper is less
            confident about (Igbo, per the caveat above).

    Returns:
        {
            "detected_language": str or None,   # "Hausa"/"Igbo"/"Yoruba"/"English"/None
            "confidence": float,
            "auto_detect_reliable": bool,        # False -> ask the nurse to confirm manually
            "raw_probabilities": dict            # all language probabilities, for debugging
        }
    """
    model = _load_detection_model()

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)

    # Sort by probability, check the top guess
    top_code = max(probs, key=probs.get)
    top_confidence = probs[top_code]

    detected_language = WHISPER_CODE_TO_LANGUAGE.get(top_code)
    reliable = detected_language is not None and top_confidence >= confidence_threshold

    if not reliable:
        logger.warning(
            f"Language auto-detection not reliable (top guess: {top_code} @ "
            f"{top_confidence:.2f}). Falling back to manual selection."
        )

    return {
        "detected_language": detected_language if reliable else None,
        "confidence": round(float(top_confidence), 3),
        "auto_detect_reliable": reliable,
        "raw_probabilities": {k: round(float(v), 3) for k, v in probs.items()},
    }
