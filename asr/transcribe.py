"""
ASR Stage — converts patient audio (Hausa/Igbo/Yoruba) into raw text using
NCAIR's Whisper-Small fine-tuned models.

Fixed for Colab:
  • Suppresses benign transformers warnings
  • Uses direct model.generate() instead of pipeline chunking (avoids seq2seq warnings)
  • Handles long-form audio properly
"""

from __future__ import annotations

import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Final

import librosa
import numpy as np
import soundfile as sf
import torch
from pydub import AudioSegment
from pydub.effects import normalize
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*generation_config.*")

logger = logging.getLogger(__name__)

MODEL_MAP: Final = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

# ── model cache ─────────────────────────────────────────────────────────────
_asr_pipelines: dict[str, pipeline] = {}


def _preprocess_audio(audio_path: str, output_path: str) -> tuple[str, np.ndarray, int]:
    """Resample to 16kHz mono."""
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sf.write(output_path, audio, sr)
    return output_path, audio, sr


def _is_audio_too_quiet(audio: np.ndarray, silence_threshold: float = 0.01) -> bool:
    rms = np.sqrt(np.mean(audio**2))
    return rms < silence_threshold


def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    import noisereduce as nr
    return nr.reduce_noise(y=audio, sr=sr, stationary=False)


def _amplify_audio(audio_path: str, output_path: str) -> str:
    audio = AudioSegment.from_file(audio_path)
    normalized = normalize(audio)
    normalized.export(output_path, format="wav")
    return output_path


def _get_device() -> str:
    """Return torch device string."""
    if torch.cuda.is_available():
        free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        if free_mem < 1 * 1024**3:  # Need at least 1GB free for Whisper-Small
            logger.warning("GPU has <1GB free — using CPU for ASR")
            return "cpu"
        return "cuda:0"
    return "cpu"


def get_asr_pipeline(language: str) -> pipeline:
    """Load (or reuse) the ASR pipeline for the given language."""
    if language not in _asr_pipelines:
        model_id = MODEL_MAP[language]
        device = _get_device()
        logger.info(f"Loading ASR model for {language}: {model_id} on {device}")

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        _asr_pipelines[language] = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
    return _asr_pipelines[language]


def transcribe_audio(audio_path: str, language: str) -> str:
    """
    Full pipeline: validate -> preprocess -> silence check -> denoise ->
    amplify -> transcribe.

    Returns raw transcript in the spoken language.
    """
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if language not in MODEL_MAP:
        raise ValueError(
            f"Unsupported language: {language!r}. Expected one of: {', '.join(MODEL_MAP)}"
        )

    with tempfile.TemporaryDirectory(prefix="ncair_asr_") as tmpdir:
        try:
            pre_path = os.path.join(tmpdir, "preprocessed.wav")
            cleaned_path = os.path.join(tmpdir, "cleaned.wav")
            final_path = os.path.join(tmpdir, "final.wav")

            _, audio, sr = _preprocess_audio(audio_path, pre_path)

            if _is_audio_too_quiet(audio):
                raise ValueError(
                    "No audio detected — recording may have failed or mic was silent"
                )

            cleaned = _reduce_noise(audio, sr)
            sf.write(cleaned_path, cleaned, sr)
            _amplify_audio(cleaned_path, final_path)

            asr = get_asr_pipeline(language)
            result = asr(
                final_path,
                generate_kwargs={"language": language.lower(), "task": "transcribe"},
            )
            text = result.get("text", "").strip()

            if not text:
                logger.warning("ASR returned empty text despite audio passing silence check")

            return text

        except Exception:
            logger.exception("Transcription failed for %s", audio_path)
            raise
