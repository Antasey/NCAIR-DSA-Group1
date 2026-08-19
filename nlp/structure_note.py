"""
LLM Structuring Stage — converts raw patient transcripts into structured
English clinical notes using N-ATLaS LLM.

Runs LOCALLY (loads the model into GPU memory once, on import) — designed
to run inside Colab where a free T4 GPU is available. This is the CORE
module of NCAIR-DSA: translation + structuring into the final clinical note.
"""

import re
import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "NCAIR1/N-ATLaS"

# ============================================================================
# MODEL LOADING (happens once, when this module is first imported)
# ============================================================================

_tokenizer = None
_model = None


def _load_model():
    """Lazy-load the model on first use, so importing this file doesn't
    immediately trigger a multi-minute download/load if it's not needed yet."""
    global _tokenizer, _model

    if _model is not None:
        return  # already loaded

    logger.info("Loading N-ATLaS tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    logger.info("Loading N-ATLaS model (this can take a few minutes on first run)...")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config
    )
    logger.info("N-ATLaS loaded successfully.")


# ============================================================================
# PROMPT
# ============================================================================

STRUCTURE_PROMPT = """You are a clinical note assistant specializing in multilingual healthcare.

Your task: Take a patient's raw speech (in Yoruba, Igbo, Hausa, or English)
and structure it into a clean, organized English clinical note.

INSTRUCTIONS:
1. If the patient spoke in Yoruba, Igbo, or Hausa, translate their meaning into clear English.
2. Extract and organize EXACTLY these four fields:
   - chief_complaint: The patient's main symptom or concern (1-2 sentences, in English)
   - duration: How long the problem has lasted (e.g., "since yesterday", "3 days", "1 week")
   - severity: How bad it is on a scale (mild, moderate, or severe)
   - history: Any other relevant details or secondary symptoms mentioned

3. CRITICAL: Use ONLY information the patient actually said. Do NOT invent or assume details.
4. If a field is not mentioned by the patient, set it to "Not mentioned".
5. Respond with ONLY valid JSON. No explanatory text before or after the JSON.

EXAMPLE:
Patient speech: "My stomach has been hurting me since yesterday, I've been vomiting too, it's very bad"
Output:
{{
  "chief_complaint": "Abdominal pain with vomiting",
  "duration": "Since yesterday",
  "severity": "severe",
  "history": "Patient reports nausea accompanying the abdominal pain"
}}

Now process this patient's speech:
Patient speech: {transcript}

Output (JSON only, no other text):"""


# ============================================================================
# LLM CALLING
# ============================================================================

def call_natlas_llm(prompt: str, max_new_tokens: int = 300) -> str:
    """Generate text using the locally loaded N-ATLaS model."""
    _load_model()

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.3,   # low temp = consistent structured output
        top_p=0.9,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id
    )

    full_text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = _tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full_text[len(prompt_text):].strip()


# ============================================================================
# JSON PARSING & VALIDATION
# ============================================================================

def extract_json_from_text(text: str):
    """Pull a JSON object out of text that may include markdown fences or extra words."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', cleaned, re.DOTALL)
    if match:
        return match.group(0)
    if cleaned.startswith("{"):
        return cleaned
    return None


def validate_structure(data: dict) -> dict:
    """Ensure all required fields exist; fill missing ones with 'Not mentioned'."""
    required_fields = ["chief_complaint", "duration", "severity", "history"]
    for field in required_fields:
        if field not in data or not data[field]:
            data[field] = "Not mentioned"
    return data


def parse_llm_output(raw_output: str) -> dict:
    """Extract, parse, and validate the JSON clinical note from raw LLM text."""
    if not raw_output:
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_error": "Empty LLM output"}

    json_str = extract_json_from_text(raw_output)
    if not json_str:
        logger.warning("No JSON found in LLM output")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_error": "No JSON in output", "_raw": raw_output[:200]}

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return validate_structure(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_error": f"JSON parse failed: {str(e)}", "_raw": json_str[:200]}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structure_note(transcript: str) -> dict:
    """
    Main function: Convert raw patient transcript -> structured clinical note.

    Args:
        transcript: Patient's raw speech (Yoruba/Igbo/Hausa/English)

    Returns:
        Dict with chief_complaint, duration, severity, history (all English).
        May include _error / _raw keys if something went wrong.
    """
    if not transcript or len(transcript.strip()) == 0:
        logger.warning("Empty transcript provided")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_error": "Empty transcript"}

    logger.info(f"Structuring note from transcript: {transcript[:50]}...")

    prompt = STRUCTURE_PROMPT.format(transcript=transcript)

    try:
        raw_output = call_natlas_llm(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_error": f"LLM call failed: {str(e)}"}

    structured = parse_llm_output(raw_output)
    logger.info("Note structuring complete")
    return structured


def structure_note_batch(transcripts: list) -> list:
    """Process multiple transcripts (useful for testing)."""
    return [structure_note(t) for t in transcripts]
