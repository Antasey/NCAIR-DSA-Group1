"""
Audio-Based Language Detection & Code-Switching Check — PRE-ASR Pipeline

Adapted from Group 1's language pipeline to integrate with MediVoice app.
Keeps all original functionality:
  • FFmpeg-free audio loading
  • Audio quality gate
  • Whole-audio language ID (Whisper)
  • Chunk-level code-switch screening
  • Nurse confirmation / override support

Patched for app integration:
  • No module-level logging config (doesn't hijack app logs)
  • No standalone UI on import
  • Returns dict format app.py expects
  • Uses "tiny" Whisper by default (faster, sufficient for lang-ID)
  • Thread-safe singleton model loading
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import whisper

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
    "en": "English",
}

TARGET_INDIGENOUS_LANGUAGES = {"Hausa", "Yoruba", "Igbo"}

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_DETECTION_MODEL = "tiny"  # patched: was "base", "tiny" is 2× faster for lang-ID

CHUNK_SECONDS = 5.0
CHUNK_OVERLAP_SECONDS = 1.0
CHUNK_CONFIDENCE_THRESHOLD = 0.35

# ============================================================
# DATA STRUCTURES (kept from original)
# ============================================================

@dataclass
class AudioQuality:
    usable: bool
    duration_seconds: float
    rms: float
    peak: float
    sample_rate: int
    reason: str = ""


@dataclass
class LanguageResult:
    detected_language: Optional[str]
    confidence: float
    auto_detect_reliable: bool
    raw_probabilities: Dict[str, float]
    code_switched: bool = False
    language_segments: Optional[List[Dict[str, Any]]] = None
    languages_seen: Optional[List[str]] = None
    indigenous_languages_seen: Optional[List[str]] = None


@dataclass
class TranscriptResult:
    transcript: str
    usable: bool
    reason: str = ""


@dataclass
class PipelineResult:
    audio_quality: Dict[str, Any]
    language: Dict[str, Any]
    transcript: Dict[str, Any]
    normalized_text: Optional[str]
    status: str
    message: str


# ============================================================
# FFMPEG-FREE AUDIO LOADER (kept from original)
# ============================================================

def load_audio_without_ffmpeg(audio_path: str) -> np.ndarray:
    """Load audio directly with soundfile; WAV is recommended."""
    if sf is None:
        raise RuntimeError("soundfile is required. Install with: pip install soundfile")
    data, sample_rate = sf.read(audio_path, always_2d=False, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        raise ValueError("Audio file contains no samples.")
    target_rate = whisper.audio.SAMPLE_RATE
    if sample_rate != target_rate:
        duration = len(data) / float(sample_rate)
        target_length = max(1, int(round(duration * target_rate)))
        old_positions = np.linspace(0.0, duration, num=len(data), endpoint=False)
        new_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
        data = np.interp(new_positions, old_positions, data).astype(np.float32)
    return data


# ============================================================
# WHISPER LANGUAGE-ID MODEL — thread-safe singleton
# ============================================================

_detection_model: whisper.Whisper | None = None
_model_lock = threading.Lock()


def load_language_id_model(model_size: str = DEFAULT_DETECTION_MODEL) -> whisper.Whisper:
    """Load Whisper once. Used ONLY for language identification."""
    global _detection_model
    if _detection_model is not None:
        return _detection_model

    with _model_lock:
        if _detection_model is not None:
            return _detection_model
        logger.info("Loading Whisper '%s' for language ID...", model_size)
        _detection_model = whisper.load_model(model_size)
        logger.info("Language detection model loaded.")
    return _detection_model


# ============================================================
# PRE-ASR STAGE 1: AUDIO QUALITY CHECK (kept from original)
# ============================================================

def check_audio_quality(
    audio_path: str,
    min_duration: float = 0.5,
    max_duration: float = 300.0,
    min_rms: float = 0.003,
    max_peak: float = 1.05,
) -> AudioQuality:
    """Basic audio-quality gate. Runs before language detection and ASR."""
    if not os.path.exists(audio_path):
        return AudioQuality(False, 0, 0, 0, 0, "Audio file does not exist.")

    if sf is None:
        logger.warning("Install soundfile for detailed audio-quality checks.")
        return AudioQuality(True, 0, 0, 0, 0, "Detailed checks skipped.")

    try:
        data, sample_rate = sf.read(audio_path, always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = data.astype(np.float32)

        if len(data) == 0:
            return AudioQuality(False, 0, 0, 0, sample_rate, "Audio has no samples.")

        duration = len(data) / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(data)) + 1e-12))
        peak = float(np.max(np.abs(data)))

        if duration < min_duration:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too short.")
        if duration > max_duration:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too long.")
        if rms < min_rms:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording is too quiet or silent.")
        if peak > max_peak:
            return AudioQuality(False, duration, rms, peak, sample_rate, "Recording appears severely clipped.")

        return AudioQuality(True, duration, rms, peak, sample_rate, "Audio quality checks passed.")

    except Exception as exc:
        logger.exception("Audio quality check failed.")
        return AudioQuality(False, 0, 0, 0, 0, f"Could not inspect audio: {exc}")


# ============================================================
# PRE-ASR STAGE 2: WHOLE-AUDIO LANGUAGE ID (kept from original)
# ============================================================

def _detect_whole_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> LanguageResult:
    """Detect the most likely language directly from RAW AUDIO."""
    model = load_language_id_model(model_size)
    audio = load_audio_without_ffmpeg(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)

    top_code = max(probs, key=probs.get)
    top_confidence = float(probs[top_code])
    detected_language = SUPPORTED_LANGUAGES.get(top_code)
    reliable = detected_language is not None and top_confidence >= confidence_threshold

    if not reliable:
        logger.warning("Language detection uncertain: %s @ %.3f. Manual confirmation required.", top_code, top_confidence)

    target_probs = {
        SUPPORTED_LANGUAGES[k]: round(float(v), 3)
        for k, v in probs.items()
        if k in SUPPORTED_LANGUAGES
    }

    return LanguageResult(
        detected_language=detected_language if reliable else None,
        confidence=round(top_confidence, 3),
        auto_detect_reliable=reliable,
        raw_probabilities=target_probs,
    )


# ============================================================
# PRE-ASR STAGE 3: CHUNK-LEVEL CODE-SWITCH SCREENING (kept)
# ============================================================

def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Analyze short audio windows before ASR. CODE-SWITCH SCREENING BASELINE."""
    model = load_language_id_model(model_size)
    audio = load_audio_without_ffmpeg(audio_path)
    total_samples = len(audio)
    sample_rate = whisper.audio.SAMPLE_RATE
    chunk_size = int(chunk_seconds * sample_rate)
    step = max(1, int((chunk_seconds - overlap_seconds) * sample_rate))

    segments = []
    for start_sample in range(0, total_samples, step):
        end_sample = min(start_sample + chunk_size, total_samples)
        if end_sample <= start_sample:
            break

        chunk = audio[start_sample:end_sample]
        if len(chunk) < int(0.75 * sample_rate):
            break

        chunk = whisper.pad_or_trim(chunk)
        mel = whisper.log_mel_spectrogram(chunk).to(model.device)
        _, probs = model.detect_language(mel)

        top_code = max(probs, key=probs.get)
        confidence = float(probs[top_code])
        language = SUPPORTED_LANGUAGES.get(top_code)

        segments.append({
            "start": round(start_sample / sample_rate, 2),
            "end": round(end_sample / sample_rate, 2),
            "language": language,
            "confidence": round(confidence, 3),
            "reliable": language is not None and confidence >= confidence_threshold,
        })

        if end_sample >= total_samples:
            break

    return segments


def infer_code_switching(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flag code-switching if multiple reliable languages appear across chunks."""
    languages = list(dict.fromkeys(
        s["language"] for s in segments if s.get("reliable") and s.get("language")
    ))
    indigenous_languages = [lang for lang in languages if lang in TARGET_INDIGENOUS_LANGUAGES]

    return {
        "code_switched": len(languages) >= 2,
        "languages_seen": languages,
        "indigenous_languages_seen": indigenous_languages,
        "segments": segments,
    }


def run_pre_asr_code_switch_screening(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> Dict[str, Any]:
    """Complete PRE-ASR code-switch screening stage."""
    try:
        segments = detect_chunk_languages(audio_path, model_size=model_size)
        return infer_code_switching(segments)
    except Exception as exc:
        logger.warning("Code-switch screening failed: %s", exc)
        return {
            "code_switched": False,
            "languages_seen": [],
            "indigenous_languages_seen": [],
            "segments": [],
            "error": str(exc),
        }


# ============================================================
# PRE-ASR STAGE 4: NURSE CONFIRMATION / OVERRIDE (kept)
# ============================================================

def confirm_language(
    detected_language: Optional[str],
    confidence: float,
    nurse_language: Optional[str] = None,
) -> Dict[str, Any]:
    """Confirm the language before ASR. Nurse-provided language always wins."""
    allowed = set(SUPPORTED_LANGUAGES.values())

    if nurse_language:
        if nurse_language not in allowed:
            raise ValueError(f"Invalid language: {nurse_language}. Choose from {sorted(allowed)}")
        return {"language": nurse_language, "source": "nurse_override", "confidence": 1.0}

    if detected_language:
        return {"language": detected_language, "source": "whisper_auto_detection", "confidence": confidence}

    return {"language": None, "source": "manual_selection_required", "confidence": confidence}


# ============================================================
# POST-ASR: TRANSCRIPT QUALITY CHECK (kept)
# ============================================================

def check_transcript_quality(transcript: str, minimum_characters: int = 3) -> TranscriptResult:
    """Basic quality check after NCAIR ASR."""
    text = " ".join((transcript or "").split())
    if not text:
        return TranscriptResult("", False, "ASR returned an empty transcript.")
    if len(text) < minimum_characters:
        return TranscriptResult(text, False, "ASR transcript is too short.")

    words = text.lower().split()
    if len(words) >= 6:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return TranscriptResult(text, False, "Transcript contains excessive token repetition.")

    return TranscriptResult(text, True, "Transcript quality checks passed.")


# ============================================================
# APP INTEGRATION: detect_audio_language()
# This is the function app.py calls.
# ============================================================

def detect_audio_language(audio_path: str) -> dict[str, Any]:
    """
    Entry point for MediVoice app.

    Runs the full PRE-ASR pipeline:
      1. Audio quality check
      2. Whole-audio language ID
      3. Chunk-level code-switch screening
      4. Build context note for N-ATLaS

    Returns dict with keys app.py expects:
      detected_language, confidence, auto_detect_reliable,
      is_code_switched, code_switch_candidates, context_note
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Audio quality
    quality = check_audio_quality(audio_path)
    if not quality.usable:
        logger.warning("Audio quality failed: %s", quality.reason)
        # Still attempt language detection — app.py handles the error display

    # 2. Whole-audio language ID
    lang_result = _detect_whole_audio_language(audio_path)

    # 3. Code-switch screening
    switch_info = run_pre_asr_code_switch_screening(audio_path)

    # 4. Build candidates list for app.py
    candidates = []
    if switch_info.get("code_switched"):
        for lang in switch_info.get("languages_seen", []):
            # Chunk screening gives us languages but not per-lang confidence.
            # We use the whole-audio confidence as a rough proxy for the primary.
            conf = lang_result.confidence if lang == lang_result.detected_language else 0.35
            candidates.append((lang, conf))

    # 5. Build context note for N-ATLaS LLM
    if switch_info.get("code_switched") and candidates:
        langs_str = ", ".join(f"{lang} ({prob:.0%})" for lang, prob in candidates)
        context_note = (
            f"Language: primarily {lang_result.detected_language or 'unknown'}, "
            f"but code-switching was detected ({langs_str}). "
            f"Translate the full meaning consistently into English "
            f"regardless of which language each part was spoken in."
        )
    else:
        context_note = (
            f"Language: {lang_result.detected_language or 'unknown'} "
            f"(confidence {lang_result.confidence:.0%}). "
            f"No code-switching detected."
        )

    # 6. Return format app.py expects
    # NEVER return None for detected_language — always fall back to a valid language
    top_language = lang_result.detected_language or "Hausa"

    return {
        "detected_language": top_language,
        "confidence": float(lang_result.confidence),
        "auto_detect_reliable": lang_result.auto_detect_reliable,
        "is_code_switched": switch_info.get("code_switched", False),
        "code_switch_candidates": candidates,
        "context_note": context_note,
        # Extras for debugging / advanced use
        "_audio_quality": asdict(quality),
        "_raw_probabilities": lang_result.raw_probabilities,
        "_languages_seen": switch_info.get("languages_seen", []),
        "_indigenous_languages_seen": switch_info.get("indigenous_languages_seen", []),
    }


# ============================================================
# FULL PIPELINE (optional — for batch/script use)
# ============================================================

def run_language_pipeline(
    audio_path: str,
    ncair_asr_fn: Callable[..., Any],
    nurse_language: Optional[str] = None,
    detector_model_size: str = DEFAULT_DETECTION_MODEL,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    run_chunk_code_switch_check: bool = True,
    normalizer_fn: Optional[Callable[..., str]] = None,
) -> PipelineResult:
    """
    Complete multilingual language pipeline.
    PRE-ASR -> ASR -> POST-ASR
    """
    # 1. Audio quality
    quality = check_audio_quality(audio_path)
    if not quality.usable:
        return PipelineResult(asdict(quality), {}, {}, None, "audio_quality_failed", quality.reason)

    # 2. Language ID
    language_result = _detect_whole_audio_language(audio_path, confidence_threshold, detector_model_size)

    # 3. Code-switch screening
    code_switch_info = {"code_switched": False, "languages_seen": [], "indigenous_languages_seen": [], "segments": []}
    if run_chunk_code_switch_check:
        code_switch_info = run_pre_asr_code_switch_screening(audio_path, model_size=detector_model_size)

    # 4. Nurse confirmation
    confirmation = confirm_language(language_result.detected_language, language_result.confidence, nurse_language=nurse_language)
    selected_language = confirmation["language"]

    if selected_language is None:
        return PipelineResult(
            asdict(quality),
            {**asdict(language_result), "confirmation": confirmation, "code_switching": code_switch_info},
            {},
            None,
            "manual_language_selection_required",
            "Language could not be trusted automatically. Select the language manually.",
        )

    # 5. ASR
    try:
        transcript = ncair_asr_fn(audio_path, language=selected_language)
    except TypeError:
        transcript = ncair_asr_fn(audio_path, selected_language)

    if isinstance(transcript, dict):
        for key in ("text", "transcript", "transcription"):
            if key in transcript:
                transcript = str(transcript[key])
                break

    # 6. Transcript quality
    transcript_result = check_transcript_quality(transcript)
    if not transcript_result.usable:
        return PipelineResult(
            asdict(quality),
            {**asdict(language_result), "confirmation": confirmation, "selected_language": selected_language, "code_switching": code_switch_info},
            asdict(transcript_result),
            None,
            "transcript_quality_failed",
            transcript_result.reason,
        )

    # 7. Optional normalization
    normalized = transcript_result.transcript
    if normalizer_fn is not None:
        try:
            normalized = str(normalizer_fn(transcript_result.transcript, language=selected_language))
        except TypeError:
            normalized = str(normalizer_fn(transcript_result.transcript, selected_language))

    return PipelineResult(
        asdict(quality),
        {**asdict(language_result), "confirmation": confirmation, "selected_language": selected_language, "code_switching": code_switch_info},
        asdict(transcript_result),
        normalized,
        "success",
        "Language pipeline completed successfully.",
    )
