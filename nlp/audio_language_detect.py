'''NCAIR Multi-Lingual Patient Intake - Language Pipeline

PRE-ASR PIPELINE:
Audio
  -> Audio Quality Check
  -> Whole-Audio Language ID
  -> Code-Switch Screening
  -> Nurse Confirmation / Override
  -> NCAIR ASR
  -> Transcript Quality Check
  -> Optional Normalization / N-ATLaS

Important:
- Whisper language ID runs on RAW AUDIO before NCAIR ASR.
- Code-switch screening also runs on RAW AUDIO before NCAIR ASR.
- Chunk-level switching is a screening baseline, not true
  word-level code-switch detection.
- The exact NCAIR ASR/N-ATLaS API is not assumed; connect your
  existing functions through the provided adapters.

'''


from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import whisper

try:
    import soundfile as sf
except ImportError:
    sf = None


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("ncair-language-pipeline")

# ============================================================
# FFmpeg-FREE AUDIO LOADER
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



SUPPORTED_LANGUAGES = {
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",      # Verify empirically with your Whisper build
    "en": "English",
}

TARGET_INDIGENOUS_LANGUAGES = {
    "Hausa",
    "Yoruba",
    "Igbo",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_DETECTION_MODEL = "base"

# Settings for chunk-level code-switch screening
CHUNK_SECONDS = 5.0
CHUNK_OVERLAP_SECONDS = 1.0
CHUNK_CONFIDENCE_THRESHOLD = 0.35


# Whisper model used only for language identification
_detection_model = None


# ============================================================
# DATA STRUCTURES
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

    # Pre-ASR code-switch information
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
# WHISPER LANGUAGE-ID MODEL
# ============================================================

def load_language_id_model(
    model_size: str = DEFAULT_DETECTION_MODEL
):
    """
    Load Whisper once.

    This Whisper model is used ONLY for language identification.
    It is not the NCAIR ASR model.
    """

    global _detection_model

    if _detection_model is None:
        logger.info(
            "Loading Whisper '%s' for language ID...",
            model_size
        )

        _detection_model = whisper.load_model(model_size)

    return _detection_model


# ============================================================
# PRE-ASR STAGE 1:
# AUDIO QUALITY CHECK
# ============================================================

def check_audio_quality(
    audio_path: str,
    min_duration: float = 0.5,
    max_duration: float = 300.0,
    min_rms: float = 0.003,
    max_peak: float = 1.05,
) -> AudioQuality:
    """
    Basic audio-quality gate.

    This runs before language detection and before ASR.
    """

    if not os.path.exists(audio_path):
        return AudioQuality(
            False,
            0,
            0,
            0,
            0,
            "Audio file does not exist."
        )

    if sf is None:
        logger.warning(
            "Install soundfile for detailed audio-quality checks."
        )

        return AudioQuality(
            True,
            0,
            0,
            0,
            0,
            "Detailed checks skipped."
        )

    try:
        data, sample_rate = sf.read(
            audio_path,
            always_2d=False
        )

        # Convert stereo/multi-channel audio to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        data = data.astype(np.float32)

        if len(data) == 0:
            return AudioQuality(
                False,
                0,
                0,
                0,
                sample_rate,
                "Audio has no samples."
            )

        duration = len(data) / float(sample_rate)

        rms = float(
            np.sqrt(
                np.mean(np.square(data)) + 1e-12
            )
        )

        peak = float(
            np.max(np.abs(data))
        )

        if duration < min_duration:
            return AudioQuality(
                False,
                duration,
                rms,
                peak,
                sample_rate,
                "Recording is too short."
            )

        if duration > max_duration:
            return AudioQuality(
                False,
                duration,
                rms,
                peak,
                sample_rate,
                "Recording is too long."
            )

        if rms < min_rms:
            return AudioQuality(
                False,
                duration,
                rms,
                peak,
                sample_rate,
                "Recording is too quiet or silent."
            )

        if peak > max_peak:
            return AudioQuality(
                False,
                duration,
                rms,
                peak,
                sample_rate,
                "Recording appears severely clipped."
            )

        return AudioQuality(
            True,
            duration,
            rms,
            peak,
            sample_rate,
            "Audio quality checks passed."
        )

    except Exception as exc:

        logger.exception(
            "Audio quality check failed."
        )

        return AudioQuality(
            False,
            0,
            0,
            0,
            0,
            f"Could not inspect audio: {exc}"
        )


# ============================================================
# PRE-ASR STAGE 2:
# WHOLE-AUDIO LANGUAGE IDENTIFICATION
# ============================================================

def detect_audio_language(
    audio_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_size: str = DEFAULT_DETECTION_MODEL,
) -> LanguageResult:
    """
    Detect the most likely language directly from RAW AUDIO.

    This happens before NCAIR ASR.
    """

    model = load_language_id_model(model_size)

    audio = load_audio_without_ffmpeg(audio_path)

    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(
        audio
    ).to(model.device)

    _, probs = model.detect_language(mel)

    top_code = max(
        probs,
        key=probs.get
    )

    top_confidence = float(
        probs[top_code]
    )

    detected_language = SUPPORTED_LANGUAGES.get(
        top_code
    )

    reliable = (
        detected_language is not None
        and top_confidence >= confidence_threshold
    )

    if not reliable:

        logger.warning(
            "Language detection uncertain: %s @ %.3f. "
            "Manual confirmation required.",
            top_code,
            top_confidence
        )

    target_probs = {
        SUPPORTED_LANGUAGES[k]: round(
            float(v),
            3
        )
        for k, v in probs.items()
        if k in SUPPORTED_LANGUAGES
    }

    return LanguageResult(
        detected_language=(
            detected_language
            if reliable
            else None
        ),

        confidence=round(
            top_confidence,
            3
        ),

        auto_detect_reliable=reliable,

        raw_probabilities=target_probs,
    )


# ============================================================
# PRE-ASR STAGE 3:
# CHUNK-LEVEL LANGUAGE ANALYSIS
# ============================================================

def detect_chunk_languages(
    audio_path: str,
    model_size: str = DEFAULT_DETECTION_MODEL,
    chunk_seconds: float = CHUNK_SECONDS,
    overlap_seconds: float = CHUNK_OVERLAP_SECONDS,
    confidence_threshold: float = CHUNK_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Analyze short audio windows before ASR.

    The purpose is to identify whether different parts of the
    recording appear to use different languages.

    This is a CODE-SWITCH SCREENING BASELINE.

    It is NOT a true word-level code-switch detector.
    """

    model = load_language_id_model(model_size)

    audio = load_audio_without_ffmpeg(audio_path)

    total_samples = len(audio)

    sample_rate = whisper.audio.SAMPLE_RATE

    chunk_size = int(
        chunk_seconds * sample_rate
    )

    step = max(
        1,
        int(
            (chunk_seconds - overlap_seconds)
            * sample_rate
        )
    )

    segments = []

    for start_sample in range(
        0,
        total_samples,
        step
    ):

        end_sample = min(
            start_sample + chunk_size,
            total_samples
        )

        if end_sample <= start_sample:
            break

        chunk = audio[
            start_sample:end_sample
        ]

        # Ignore extremely short final chunks
        if len(chunk) < int(
            0.75 * sample_rate
        ):
            break

        chunk = whisper.pad_or_trim(
            chunk
        )

        mel = whisper.log_mel_spectrogram(
            chunk
        ).to(model.device)

        _, probs = model.detect_language(
            mel
        )

        top_code = max(
            probs,
            key=probs.get
        )

        confidence = float(
            probs[top_code]
        )

        language = SUPPORTED_LANGUAGES.get(
            top_code
        )

        segments.append(
            {
                "start": round(
                    start_sample / sample_rate,
                    2
                ),

                "end": round(
                    end_sample / sample_rate,
                    2
                ),

                "language": language,

                "confidence": round(
                    confidence,
                    3
                ),

                "reliable": (
                    language is not None
                    and confidence >= confidence_threshold
                ),
            }
        )

        if end_sample >= total_samples:
            break

    return segments


# ============================================================
# PRE-ASR STAGE 4:
# CODE-SWITCH INFERENCE
# ============================================================

def infer_code_switching(
    segments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determine whether multiple reliable languages appear
    across the audio chunks.

    If two or more reliable languages occur in different
    chunks, flag the recording as potentially code-switched.
    """

    languages = list(
        dict.fromkeys(
            s["language"]
            for s in segments
            if s.get("reliable")
            and s.get("language")
        )
    )

    indigenous_languages = [
        language
        for language in languages
        if language in TARGET_INDIGENOUS_LANGUAGES
    ]

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
    """
    Complete PRE-ASR code-switch screening stage.

    RAW AUDIO
        -> short audio chunks
        -> language ID for each chunk
        -> compare languages
        -> possible code-switch flag
    """

    try:

        segments = detect_chunk_languages(
            audio_path,
            model_size=model_size,
        )

        return infer_code_switching(
            segments
        )

    except Exception as exc:

        logger.warning(
            "Code-switch screening failed: %s",
            exc
        )

        return {
            "code_switched": False,
            "languages_seen": [],
            "indigenous_languages_seen": [],
            "segments": [],
            "error": str(exc),
        }


# ============================================================
# PRE-ASR STAGE 5:
# NURSE CONFIRMATION / OVERRIDE
# ============================================================

def confirm_language(
    detected_language: Optional[str],
    confidence: float,
    nurse_language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Confirm the language before ASR.

    A nurse-provided language always takes priority.
    """

    allowed = set(
        SUPPORTED_LANGUAGES.values()
    )

    # Human override wins
    if nurse_language:

        if nurse_language not in allowed:

            raise ValueError(
                f"Invalid language: {nurse_language}. "
                f"Choose from {sorted(allowed)}"
            )

        return {
            "language": nurse_language,
            "source": "nurse_override",
            "confidence": 1.0,
        }

    # Automatic detection
    if detected_language:

        return {
            "language": detected_language,
            "source": "whisper_auto_detection",
            "confidence": confidence,
        }

    # No reliable result
    return {
        "language": None,
        "source": "manual_selection_required",
        "confidence": confidence,
    }


# ============================================================
# ASR STAGE
# ============================================================

def run_ncair_asr(
    audio_path: str,
    language: str,
    ncair_asr_fn: Callable[..., Any],
) -> str:
    """
    Adapter for the existing NCAIR ASR.

    Replace the callable with the group's actual ASR function.
    """

    try:

        result = ncair_asr_fn(
            audio_path,
            language=language
        )

    except TypeError:

        result = ncair_asr_fn(
            audio_path,
            language
        )

    if isinstance(result, dict):

        for key in (
            "text",
            "transcript",
            "transcription"
        ):

            if key in result:
                return str(
                    result[key]
                )

    return str(result)


# ============================================================
# POST-ASR STAGE:
# TRANSCRIPT QUALITY CHECK
# ============================================================

def check_transcript_quality(
    transcript: str,
    minimum_characters: int = 3,
) -> TranscriptResult:
    """
    Basic quality check after NCAIR ASR.
    """

    text = " ".join(
        (transcript or "").split()
    )

    if not text:

        return TranscriptResult(
            "",
            False,
            "ASR returned an empty transcript."
        )

    if len(text) < minimum_characters:

        return TranscriptResult(
            text,
            False,
            "ASR transcript is too short."
        )

    words = text.lower().split()

    if len(words) >= 6:

        unique_ratio = (
            len(set(words))
            / len(words)
        )

        if unique_ratio < 0.25:

            return TranscriptResult(
                text,
                False,
                "Transcript contains excessive token repetition."
            )

    return TranscriptResult(
        text,
        True,
        "Transcript quality checks passed."
    )


# ============================================================
# OPTIONAL NORMALIZATION / N-ATLaS
# ============================================================

def normalize_transcript(
    transcript: str,
    language: str,
    normalizer_fn: Optional[
        Callable[..., str]
    ] = None,
) -> str:
    """
    Optional hook for translation, normalization,
    or N-ATLaS processing.
    """

    if normalizer_fn is None:
        return transcript

    try:

        return str(
            normalizer_fn(
                transcript,
                language=language
            )
        )

    except TypeError:

        return str(
            normalizer_fn(
                transcript,
                language
            )
        )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_language_pipeline(
    audio_path: str,
    ncair_asr_fn: Callable[..., Any],
    nurse_language: Optional[str] = None,
    detector_model_size: str = DEFAULT_DETECTION_MODEL,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    run_chunk_code_switch_check: bool = True,
    normalizer_fn: Optional[
        Callable[..., str]
    ] = None,
) -> PipelineResult:
    """
    Complete multilingual language pipeline.

    ==========================================================
    PRE-ASR
    ==========================================================

    1. Audio quality check
    2. Whole-audio language identification
    3. Chunk-level code-switch screening
    4. Nurse confirmation / override

    ==========================================================
    ASR
    ==========================================================

    5. NCAIR ASR

    ==========================================================
    POST-ASR
    ==========================================================

    6. Transcript quality check
    7. Optional normalization / N-ATLaS
    """

    # --------------------------------------------------------
    # PRE-ASR STAGE 1: AUDIO QUALITY
    # --------------------------------------------------------

    quality = check_audio_quality(
        audio_path
    )

    if not quality.usable:

        return PipelineResult(
            asdict(quality),
            {},
            {},
            None,
            "audio_quality_failed",
            quality.reason
        )

    # --------------------------------------------------------
    # PRE-ASR STAGE 2: WHOLE-AUDIO LANGUAGE ID
    # --------------------------------------------------------

    language_result = detect_audio_language(
        audio_path,
        confidence_threshold=confidence_threshold,
        model_size=detector_model_size,
    )

    # --------------------------------------------------------
    # PRE-ASR STAGE 3: CODE-SWITCH SCREENING
    # --------------------------------------------------------

    code_switch_info = {
        "code_switched": False,
        "languages_seen": [],
        "indigenous_languages_seen": [],
        "segments": [],
    }

    if run_chunk_code_switch_check:

        code_switch_info = (
            run_pre_asr_code_switch_screening(
                audio_path,
                model_size=detector_model_size,
            )
        )

    # --------------------------------------------------------
    # PRE-ASR STAGE 4:
    # NURSE CONFIRMATION / OVERRIDE
    # --------------------------------------------------------

    confirmation = confirm_language(
        language_result.detected_language,
        language_result.confidence,
        nurse_language=nurse_language,
    )

    selected_language = confirmation[
        "language"
    ]

    # If automatic language detection is uncertain
    # and nurse has not supplied an override
    if selected_language is None:

        return PipelineResult(

            asdict(quality),

            {
                **asdict(language_result),

                "confirmation": confirmation,

                "code_switching": code_switch_info,
            },

            {},

            None,

            "manual_language_selection_required",

            (
                "Language could not be trusted automatically. "
                "Select the language manually."
            ),
        )

    # --------------------------------------------------------
    # ASR STAGE:
    # NCAIR TRANSCRIPTION
    # --------------------------------------------------------

    transcript = run_ncair_asr(
        audio_path,
        selected_language,
        ncair_asr_fn,
    )

    # --------------------------------------------------------
    # POST-ASR STAGE:
    # TRANSCRIPT QUALITY
    # --------------------------------------------------------

    transcript_result = check_transcript_quality(
        transcript
    )

    if not transcript_result.usable:

        return PipelineResult(

            asdict(quality),

            {
                **asdict(language_result),

                "confirmation": confirmation,

                "selected_language": selected_language,

                "code_switching": code_switch_info,
            },

            asdict(
                transcript_result
            ),

            None,

            "transcript_quality_failed",

            transcript_result.reason,
        )

    # --------------------------------------------------------
    # OPTIONAL NORMALIZATION / N-ATLaS
    # --------------------------------------------------------

    normalized = normalize_transcript(
        transcript_result.transcript,
        selected_language,
        normalizer_fn=normalizer_fn,
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return PipelineResult(

        asdict(quality),

        {
            **asdict(language_result),

            "confirmation": confirmation,

            "selected_language": selected_language,

            "code_switching": code_switch_info,
        },

        asdict(
            transcript_result
        ),

        normalized,

        "success",

        "Language pipeline completed successfully.",
    )


# ============================================================
# EXAMPLE NCAIR ASR ADAPTER
# ============================================================

def example_ncair_asr(
    audio_path: str,
    language: str
) -> str:
    """
    Replace this with the EXISTING NCAIR ASR function.

    Do not use this placeholder in production.
    """

    raise NotImplementedError(
        "Connect example_ncair_asr() "
        "to your existing NCAIR ASR implementation."
    )


# ============================================================
# COMMAND-LINE TEST
# ============================================================


# ============================================================
# SIMPLE GRADIO UI
# ============================================================

def launch_ui():
    """
    Simple standalone UI for the language component.
    It runs only the user's pre-ASR language stages:
      Audio Quality -> Language ID -> Code-Switch Screening
      -> Nurse Confirmation
    It does not require the group's NCAIR ASR implementation.
    """
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit(
            "Gradio is required for the UI. Install it with:\n"
            "pip install gradio"
        )

    def process_audio(audio_path, nurse_language, model_size, threshold):
        if not audio_path:
            return (
                "No audio selected.",
                "",
                "",
                "",
            )

        try:
            # Stage 1: audio quality
            quality = check_audio_quality(audio_path)

            if not quality.usable:
                return (
                    "❌ Audio quality check failed",
                    json.dumps(asdict(quality), indent=2),
                    "",
                    "",
                )

            # Stage 2: whole-audio language ID
            language = detect_audio_language(
                audio_path,
                confidence_threshold=float(threshold),
                model_size=model_size,
            )

            # Stage 3: code-switch screening
            code_switch = run_pre_asr_code_switch_screening(
                audio_path,
                model_size=model_size,
            )

            # Stage 4: nurse confirmation / override
            confirmation = confirm_language(
                language.detected_language,
                language.confidence,
                nurse_language=nurse_language or None,
            )

            status = (
                "✅ PRE-ASR CHECKS PASSED"
                if confirmation["language"]
                else "⚠️ MANUAL LANGUAGE SELECTION REQUIRED"
            )

            summary = (
                f"{status}\n\n"
                f"Selected language: {confirmation['language'] or 'None'}\n"
                f"Detection confidence: {language.confidence:.3f}\n"
                f"Confirmation source: {confirmation['source']}\n"
                f"Possible code-switch: "
                f"{'YES' if code_switch['code_switched'] else 'NO'}\n"
                f"Languages seen: "
                f"{', '.join(code_switch['languages_seen']) or 'None'}"
            )

            quality_text = json.dumps(
                asdict(quality), indent=2
            )

            language_text = json.dumps(
                asdict(language), indent=2
            )

            switch_text = json.dumps(
                code_switch, indent=2
            )

            return (
                summary,
                quality_text,
                language_text,
                switch_text,
            )

        except Exception as exc:
            logger.exception("UI processing failed.")
            message = str(exc)
            if "Format not recognised" in message or "Error opening" in message:
                message = "This FFmpeg-free demo expects a WAV file. Please record with the microphone or upload WAV audio."
            return (f"❌ Error: {message}", "", "", "")

    with gr.Blocks(
        title="NCAIR Multi-Lingual Patient Intake - Language Pipeline"
    ) as demo:

        gr.Markdown(
            """
            # 🏥 NCAIR Multi-Lingual Patient Intake
            ## Language Detection & Pre-ASR Pipeline

            **Pipeline:**  
            🎙️ Audio → 🔊 Quality Check → 🌍 Language ID →
            🔄 Code-Switch Screening → 👩‍⚕️ Nurse Confirmation

            This interface demonstrates the **language component only**.
            NCAIR ASR is connected later by the main group pipeline.
            """
        )

        with gr.Row():
            with gr.Column():
                audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Patient Audio (WAV recommended — no FFmpeg required)",
                )

                model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large"],
                    value="base",
                    label="Whisper Language-ID Model",
                )

                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.50,
                    step=0.05,
                    label="Language Confidence Threshold",
                )

                nurse_language = gr.Dropdown(
                    choices=[
                        "",
                        "English",
                        "Hausa",
                        "Igbo",
                        "Yoruba",
                    ],
                    value="",
                    label="Nurse Language Override (Optional)",
                )

                run_button = gr.Button(
                    "Run Language Analysis",
                    variant="primary",
                )

            with gr.Column():
                summary = gr.Textbox(
                    label="Result",
                    lines=8,
                )

                quality_output = gr.Code(
                    label="1. Audio Quality",
                    language="json",
                )

                language_output = gr.Code(
                    label="2. Whole-Audio Language ID",
                    language="json",
                )

                switch_output = gr.Code(
                    label="3. Code-Switch Screening",
                    language="json",
                )

        run_button.click(
            fn=process_audio,
            inputs=[
                audio,
                nurse_language,
                model,
                threshold,
            ],
            outputs=[
                summary,
                quality_output,
                language_output,
                switch_output,
            ],
        )

        gr.Markdown(
            """
            ### Note
            Code-switch screening is a **chunk-level screening baseline**,
            not a true word-level code-switch detector.
            """
        )

    demo.launch()


if __name__ == "__main__":
    launch_ui()
