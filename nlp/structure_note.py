"""
structure.py

Loads N-ATLaS locally (GGUF, via llama-cpp-python) and turns an ASR
transcript into a structured English clinical note.

Model loading is a lazy singleton behind get_llm(), so extract.py can
import and reuse the *same* in-memory model instead of loading a second
copy — important on an 8GB machine where you don't want two model
instances resident at once.
"""

import gc
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Optional

from llama_cpp import Llama

logger = logging.getLogger(__name__)

# --- Configuration ----------------------------------------------------
# Swap the quant file here. Q4_K_M is the default recommendation if
# N-ATLaS is loaded/unloaded sequentially relative to ASR; drop to
# Q3_K_M/Q3_K_S if ASR and N-ATLaS ever need to be resident together.

MODEL_PATH = Path("models/N-ATLaS.Q4_K_M.gguf")
N_CTX = 3072          # clinical notes don't need more; keeps KV cache small
N_THREADS = 6          # set to the target machine's physical core count
N_GPU_LAYERS = 0       # 0 = CPU only; raise if an iGPU is available

# --- Singleton model handle ---------------------------------------------

_llm: Optional[Llama] = None
_llm_lock = Lock()


def get_llm() -> Llama:
    """Return a shared Llama instance, loading it on first use.

    Both structure.py and extract.py call this so only one copy of
    N-ATLaS is ever resident in RAM at a time.
    """
    global _llm
    with _llm_lock:
        if _llm is None:
            logger.info("Loading N-ATLaS from %s", MODEL_PATH)
            _llm = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=N_CTX,
                n_threads=N_THREADS,
                n_gpu_layers=N_GPU_LAYERS,
                verbose=False,
            )
        return _llm


def unload_llm() -> None:
    """Release N-ATLaS from RAM.

    Call this once a note has been reviewed/saved, if the freed memory
    is needed elsewhere (e.g. for the ASR stage) before the next patient.
    """
    global _llm
    with _llm_lock:
        if _llm is not None:
            del _llm
            _llm = None
            gc.collect()
            logger.info("N-ATLaS unloaded")


# --- Prompting ----------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a clinical scribe assistant. You are given a transcript of a "
    "patient describing their symptoms, originally spoken in Hausa, Igbo, "
    "Yoruba, or English and already transcribed to text. Rewrite it as a "
    "structured clinical intake note in English. Do not add symptoms, "
    "durations, or details that are not stated or clearly implied in the "
    "transcript. Do not guess at a diagnosis or cause. Respond with ONLY a "
    "JSON object matching this schema, no commentary before or after it:\n"
    "{\n"
    '  "chief_complaint": string,\n'
    '  "history_of_present_illness": string,\n'
    '  "symptoms": [string],\n'
    '  "duration": string,\n'
    '  "additional_notes": string\n'
    "}"
)


def _build_prompt(transcript: str, language: Optional[str]) -> str:
    lang_line = f"Source language: {language}\n" if language else ""
    return (
        f"{lang_line}Transcript:\n{transcript}\n\n"
        "Return only the JSON object described in your instructions."
    )


def structure_note(transcript: str, language: Optional[str] = None) -> dict:
    """Turn a raw ASR transcript into a structured clinical note.

    Returns a dict matching the schema in SYSTEM_PROMPT, plus the raw
    transcript and source language for the audit trail. If the model
    doesn't return valid JSON, the raw text is preserved in
    'additional_notes' rather than dropping the note entirely — the
    doctor's review step still gets something usable to edit.
    """
    llm = get_llm()
    prompt = _build_prompt(transcript, language)

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=600,
    )

    raw = result["choices"][0]["message"]["content"].strip()

    try:
        note = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("N-ATLaS did not return valid JSON; storing raw text")
        note = {
            "chief_complaint": "",
            "history_of_present_illness": "",
            "symptoms": [],
            "duration": "",
            "additional_notes": raw,
        }

    note["raw_transcript"] = transcript
    note["source_language"] = language
    return note
