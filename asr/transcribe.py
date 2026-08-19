"""
ASR stage — loads the matching NCAIR Whisper-Small fine-tune per language
and transcribes patient audio to raw text.
"""
from transformers import pipeline

MODEL_MAP = {
    "Hausa": "NCAIR1/Hausa-ASR",
    "Igbo": "NCAIR1/Igbo-ASR",
    "Yoruba": "NCAIR1/Yoruba-ASR",
}

_pipelines = {}

def get_pipeline(language: str):
    if language not in _pipelines:
        model_id = MODEL_MAP[language]
        _pipelines[language] = pipeline("automatic-speech-recognition", model=model_id)
    return _pipelines[language]

def transcribe_audio(audio_path: str, language: str) -> str:
    asr = get_pipeline(language)
    result = asr(audio_path)
    return result["text"]
