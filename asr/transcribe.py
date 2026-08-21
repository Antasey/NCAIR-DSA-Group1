"""
ASR stage — loads the matching NCAIR Whisper-Small fine-tune per language
and transcribes patient audio to raw text.
"""
import torch
from transformers import pipeline

MODEL_MAP = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

_pipelines = {}

def get_pipeline(language: str):
    # Load the model only once and reuse it
    if language not in _pipelines:
        model_id = MODEL_MAP[language]

        # Use the GPU if available; otherwise use the CPU
        device = 0 if torch.cuda.is_available() else -1

        # Initialize pipeline with 30s chunking for long audio
        _pipelines[language] = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            device=device
        )
    return _pipelines[language]

def transcribe_audio(audio_path: str, language: str) -> str:
    # Retrieve the correct pipeline for the requested language
    asr = get_pipeline(language)

    # Pass the audio file to the ASR pipeline for transcription
    result = asr(audio_path)

    # Return the raw spoken transcript
    return result["text"]