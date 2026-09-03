"""
Audio-Based Language Detection & Code-Switching Check — runs BEFORE ASR,
directly on raw audio.

ORDER (deliberately): Whisper's full language probability distribution is
computed once, and the code-switching check runs FIRST against that
distribution — before a single "primary language" is finalized. This
matters because collapsing straight to a top-1 guess throws away the
signal a code-switching check actually needs: if two languages both score
meaningfully high, that IS the code-switching signal, sitting right there
in the distribution. Checking after top-1 has already been picked means
working with less information than is actually available.

Flags code-switching but does NOT block the pipeline — detection continues
regardless, and the flag is carried forward into language_context for
N-ATLaS, so the LLM knows to expect mixed input rather than being
surprised by it mid-transcript.

Uses the small/base OpenAI Whisper model (not the NCAIR fine-tunes) purely
for its language-ID head — loaded once, separate from the NCAIR ASR models
used for actual transcription.

*** IMPORTANT — VERIFY BEFORE RELYING ON THIS IN A DEMO ***
Whisper's standard language list includes Yoruba ('yo') and Hausa ('ha').
Igbo ('ig') is NOT confirmed to be a standard Whisper language. The
confidence threshold and manual-selection fallback below exist specifically
to protect against silently mislabeling Igbo audio — this needs to be
tested empirically with real Igbo audio before being relied on live.
"""

import logging
import whisper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WHISPER_CODE_TO_LANGUAGE = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",  # unconfirmed as a standard Whisper code — verify empirically
    "en": "English",
}

# Threshold for flagging code-switching: if two or more of our target
# languages each score above this in Whisper's probability distribution,
# the audio is treated as likely mixed-language.
CODE_SWITCH_THRESHOLD = 0.15

_detection_model = None


def _load_detection_model(model_size: str = "base"):
    """Lazy-load the small Whisper model used only for language ID."""
    global _detection_model
    if _detection_model is None:
        logger.info(f"Loading Whisper '{model_size}' model for language detection...")
        _detection_model = whisper.load_model(model_size)
    return _detection_model


def _get_language_probabilities(audio_path: str) -> dict:
    """Run Whisper's language-ID head, return the full probability
    distribution across ALL languages Whisper recognizes (not just ours)."""
    model = _load_detection_model()
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    return probs


def check_code_switching(probs: dict) -> dict:
    """
    STEP 1 (runs first): check the FULL distribution for signs of
    code-switching, before any single language has been finalized.

    Args:
        probs: full Whisper language probability dict (all languages)

    Returns:
        {
            "is_code_switched": bool,
            "candidate_languages": [(language, probability), ...],  # our
                target languages that scored above CODE_SWITCH_THRESHOLD,
                sorted highest first
        }
    """
    candidates = []
    for code, language_name in WHISPER_CODE_TO_LANGUAGE.items():
        score = probs.get(code, 0.0)
        if score >= CODE_SWITCH_THRESHOLD:
            candidates.append((language_name, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    is_code_switched = len(candidates) >= 2

    if is_code_switched:
        logger.info(f"Code-switching detected: {candidates}")

    return {
        "is_code_switched": is_code_switched,
        "candidate_languages": candidates,
    }


def detect_audio_language(audio_path: str, confidence_threshold: float = 0.5) -> dict:
    """
    Full detection pipeline, in order:
      1. Get Whisper's full language probability distribution
      2. Check for code-switching against that FULL distribution (before
         picking a single "primary" language)
      3. THEN finalize the primary/top language for ASR model selection

    Flags code-switching but always continues — never blocks the pipeline.

    Returns:
        {
            "detected_language": str or None,     # top language, for ASR model selection
            "confidence": float,                    # top language's confidence
            "auto_detect_reliable": bool,           # False -> nurse confirms manually
            "is_code_switched": bool,               # flagged, does not block
            "code_switch_candidates": [(lang, prob), ...],
            "context_note": str,                    # ready to inject into N-ATLaS prompt
            "raw_probabilities": dict,              # full distribution, for debugging
        }
    """
    probs = _get_language_probabilities(audio_path)

    # STEP 1: code-switch check runs FIRST, against the full distribution
    switch_result = check_code_switching(probs)

    # STEP 2: NOW finalize the primary language (top-1, our target languages only)
    our_language_probs = {
        WHISPER_CODE_TO_LANGUAGE[code]: probs.get(code, 0.0)
        for code in WHISPER_CODE_TO_LANGUAGE
    }
    top_language = max(our_language_probs, key=our_language_probs.get)
    top_confidence = our_language_probs[top_language]

    reliable = top_confidence >= confidence_threshold

    if not reliable:
        logger.warning(
            f"Language auto-detection not reliable (top guess: {top_language} @ "
            f"{top_confidence:.2f}). Falling back to manual selection."
        )

    # Build the context note for N-ATLaS — includes the code-switch flag
    # regardless of confidence, since the LLM should know either way
    if switch_result["is_code_switched"]:
        langs_str = ", ".join(f"{lang} ({prob:.0%})" for lang, prob in switch_result["candidate_languages"])
        context_note = (
            f"Language: primarily {top_language}, but code-switching was detected "
            f"({langs_str}). Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in."
        )
    else:
        context_note = f"Language: {top_language} (confidence {top_confidence:.0%}). No code-switching detected."

    return {
        "detected_language": top_language if reliable else None,
        "confidence": round(float(top_confidence), 3),
        "auto_detect_reliable": reliable,
        "is_code_switched": switch_result["is_code_switched"],
        "code_switch_candidates": switch_result["candidate_languages"],
        "context_note": context_note,
        "raw_probabilities": {k: round(float(v), 3) for k, v in probs.items()},
    }
