"""
LLM Structuring Stage — converts raw patient transcripts into structured
English clinical notes using N-ATLaS LLM.

Loaded via Hugging Face transformers with 4-bit quantization.
Run `python setup_models.py` BEFORE starting the app to download the model.

Colab-specific fixes:
  • low_cpu_mem_usage=True to prevent RAM blow-up during loading
  • Graceful fallback to CPU if GPU OOM
  • Clear error messages if the model can't load
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Final

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

LOCAL_MODEL_DIR: Final = Path("models/N-ATLaS")
N_CTX: Final = 3072
MAX_RETRIES: Final = 2

# ── model singleton ─────────────────────────────────────────────────────────
_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None = None
_model_lock = Lock()


def _verify_model_download() -> None:
    """Check that the model was fully downloaded."""
    if not LOCAL_MODEL_DIR.exists():
        raise RuntimeError(
            f"Model directory not found: {LOCAL_MODEL_DIR}\n"
            f"Please run: python setup_models.py"
        )

    required_files = ["config.json", "tokenizer.json"]
    missing = [f for f in required_files if not (LOCAL_MODEL_DIR / f).exists()]
    if missing:
        raise RuntimeError(
            f"Model download appears incomplete. Missing: {missing}\n"
            f"Please re-run: python setup_models.py"
        )

    # Check for model weights
    has_weights = any(
        (LOCAL_MODEL_DIR / f).exists() 
        for f in ["model.safetensors", "pytorch_model.bin", "model-00001-of-00002.safetensors"]
    )
    if not has_weights:
        raise RuntimeError(
            f"No model weights found in {LOCAL_MODEL_DIR}\n"
            f"Please re-run: python setup_models.py"
        )


def _load_model() -> None:
    """Lazy-load the transformers model on first use."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return

    with _model_lock:
        if _model is not None and _tokenizer is not None:
            return

        _verify_model_download()

        logger.info("Loading N-ATLaS tokenizer from %s ...", LOCAL_MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        logger.info("Loading N-ATLaS model... This may take 2–5 minutes on first load.")

        # Try GPU first, fall back to CPU if OOM
        load_errors = []

        if torch.cuda.is_available():
            try:
                logger.info("Attempting GPU load with 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                _model = AutoModelForCausalLM.from_pretrained(
                    LOCAL_MODEL_DIR,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float16,
                )
                logger.info("✓ N-ATLaS loaded on GPU with 4-bit quantization.")
                return
            except Exception as e:
                load_errors.append(f"GPU 4-bit failed: {e}")
                logger.warning("GPU 4-bit load failed: %s", e)

                # Clear GPU cache before retry
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                try:
                    logger.info("Retrying GPU with 8-bit quantization...")
                    bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
                    _model = AutoModelForCausalLM.from_pretrained(
                        LOCAL_MODEL_DIR,
                        quantization_config=bnb_config_8bit,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    logger.info("✓ N-ATLaS loaded on GPU with 8-bit quantization.")
                    return
                except Exception as e2:
                    load_errors.append(f"GPU 8-bit failed: {e2}")
                    logger.warning("GPU 8-bit load failed: %s", e2)
                    torch.cuda.empty_cache()

        # Fallback to CPU
        try:
            logger.warning("Falling back to CPU load (slower but more stable)...")
            _model = AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_DIR,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info("✓ N-ATLaS loaded on CPU.")
        except Exception as e:
            load_errors.append(f"CPU failed: {e}")
            logger.error("All model loading attempts failed:")
            for err in load_errors:
                logger.error("  - %s", err)
            raise RuntimeError(
                f"Failed to load N-ATLaS model. Tried GPU (4-bit, 8-bit) and CPU.\n"
                f"Last error: {e}\n"
                f"If on Colab free tier, the model may be too large. "
                f"Consider using a smaller model or upgrading to Colab Pro."
            )


def get_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Return the shared (model, tokenizer) tuple."""
    _load_model()
    if _model is None or _tokenizer is None:
        raise RuntimeError("Failed to load N-ATLaS model")
    return _model, _tokenizer


def generate_text(
    prompt: str,
    max_new_tokens: int = 750,
    temperature: float = 0.3,
) -> str:
    """
    Generate text using the loaded N-ATLaS model.
    NEVER returns an empty string — raises RuntimeError if empty.
    """
    model, tokenizer = get_model()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=N_CTX,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    if not text:
        raise RuntimeError("LLM returned an empty response — possible generation failure.")

    return text


# ============================================================================
# PROMPT
# ============================================================================

STRUCTURE_PROMPT = """You are a clinical note assistant specializing in multilingual healthcare.

{language_context}

Your task: Take a patient's raw speech and write it up as a clear, descriptive English clinical note — the way a doctor would document it in a patient's chart, not a clipped summary.

INSTRUCTIONS:
1. If the patient spoke in Yoruba, Igbo, or Hausa, translate their meaning into clear, natural English.
2. Write and organize EXACTLY these five fields:

   FIELDS 1-4 — GROUNDED IN ONLY WHAT THE PATIENT SAID:
   - chief_complaint: Describe the patient's main symptom or concern in detail.
   - duration: Describe the timeline in full.
   - severity: Describe how the patient characterized the severity.
   - history: Write a full paragraph covering any secondary symptoms, prior episodes, medications already tried, or other relevant context.

   FIELD 5 — MAY REASON BEYOND THE LITERAL STATEMENT, BUT STAY CONSERVATIVE:
   - possible_recommendations: Offer gentle, general considerations for the doctor. This is a NUDGE, NOT a diagnosis. Phrase everything as open possibilities. Keep this brief — 2-3 sentences. If too vague, write "No specific considerations suggested — insufficient detail."

3. ABSOLUTE RULE FOR FIELDS 1-4 — DO NOT HALLUCINATE: Every sentence must be traceable to something the patient actually said.
4. If a field was not mentioned, write "Not mentioned by patient".
5. Fields 1-4 must not contain clinical interpretations or diagnoses.
6. Respond with ONLY valid JSON. No explanatory text before or after the JSON.

EXAMPLE:
Patient speech: "My stomach has been hurting me since yesterday, I've been vomiting too, it's very bad, I can't even eat anything"
Output:
{{
  "chief_complaint": "The patient reports abdominal pain accompanied by vomiting. The pain is significant enough that the patient has been unable to eat since symptoms began.",
  "duration": "Symptoms began yesterday and have persisted since onset.",
  "severity": "The patient describes the pain as very bad, and it has been severe enough to prevent normal eating, indicating a severe presentation.",
  "history": "Not mentioned by patient.",
  "possible_recommendations": "Given the combination of abdominal pain and vomiting, general gastrointestinal causes may be worth considering. The doctor may wish to evaluate hydration status."
}}

Now process this patient's speech:
Patient speech: {transcript}

Output (JSON only, no other text):"""


# ============================================================================
# JSON PARSING & VALIDATION
# ============================================================================

def _extract_json_from_text(text: str) -> str | None:
    cleaned = text.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


def _validate_structure(data: dict) -> dict:
    required = ["chief_complaint", "duration", "severity", "history"]
    for field in required:
        if field not in data or not data[field]:
            data[field] = "Not mentioned by patient"
    if "possible_recommendations" not in data or not data["possible_recommendations"]:
        data["possible_recommendations"] = "No specific considerations suggested — insufficient detail."
    return data


def _parse_llm_output(raw_output: str) -> dict:
    if not raw_output:
        return _error_result("Empty LLM output")

    json_str = _extract_json_from_text(raw_output)
    if not json_str:
        logger.warning("No JSON found in LLM output")
        return _error_result("No JSON in output", raw_output[:500])

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return _validate_structure(data)
    except json.JSONDecodeError as e:
        logger.error("JSON decode error: %s", e)
        return _error_result(f"JSON parse failed: {e}", json_str[:500])


def _error_result(msg: str, raw: str = "") -> dict:
    return {
        "chief_complaint": "",
        "duration": "",
        "severity": "",
        "history": "",
        "possible_recommendations": "",
        "_error": msg,
        "_raw": raw,
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structure_note(transcript: str, language_context: str = "") -> dict:
    if not transcript or not transcript.strip():
        logger.warning("Empty transcript provided")
        return _error_result("Empty transcript")

    logger.info("Structuring note from transcript: %s...", transcript[:60])

    prompt = STRUCTURE_PROMPT.format(
        transcript=transcript.strip(),
        language_context=language_context
        or "Language: not pre-detected — determine from the transcript itself.",
    )

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_output = generate_text(prompt, max_new_tokens=750, temperature=0.1)
            structured = _parse_llm_output(raw_output)

            if "_error" in structured:
                last_error = structured["_error"]
                logger.warning("Attempt %d: parse error — %s", attempt, last_error)
                continue

            main_fields = [structured.get(f, "") for f in ["chief_complaint", "duration", "severity", "history"]]
            if all(f in ("", "Not mentioned by patient") for f in main_fields):
                raise RuntimeError("All clinical fields are empty after structuring")

            logger.info("Note structuring complete (attempt %d)", attempt)
            return structured

        except Exception as e:
            last_error = str(e)
            logger.exception("LLM call failed (attempt %d)", attempt)

    logger.error("All %d attempts failed. Last error: %s", MAX_RETRIES, last_error)
    return _error_result(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")


def structure_note_batch(transcripts: list[tuple[str, str]]) -> list[dict]:
    return [structure_note(t, ctx) for t, ctx in transcripts]
