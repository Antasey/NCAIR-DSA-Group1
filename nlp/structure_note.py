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

{language_context}

Your task: Take a patient's raw speech (in Yoruba, Igbo, Hausa, or English)
and write it up as a clear, descriptive English clinical note — the way a
doctor would document it in a patient's chart, not a clipped summary.

INSTRUCTIONS:
1. If the patient spoke in Yoruba, Igbo, or Hausa, translate their meaning into clear, natural English.
2. Write and organize EXACTLY these five fields:

   FIELDS 1-4 — GROUNDED IN ONLY WHAT THE PATIENT SAID:
   - chief_complaint: Describe the patient's main symptom or concern in detail —
     what it is, where it's located if mentioned, how it feels if described
     (sharp, dull, constant, comes and goes, etc.). Write 2-4 full sentences,
     weaving together everything the patient said about their main problem.
   - duration: Describe the timeline in full — when it started, whether it's
     gotten better/worse/stayed the same, if the patient mentioned any pattern.
     Write in a full sentence, not just a short phrase.
   - severity: Describe how the patient characterized the severity, including
     any impact they mentioned (e.g. "unable to sleep", "can't eat"). State
     the overall level (mild, moderate, or severe) but explain the reasoning
     in a full sentence, not just the single word.
   - history: Write a full paragraph covering any secondary symptoms, prior
     episodes, medications already tried, or other relevant context the
     patient mentioned — even in passing.

   FIELD 5 — MAY REASON BEYOND THE LITERAL STATEMENT, BUT STAY CONSERVATIVE:
   - possible_recommendations: Offer gentle, general considerations for the
     doctor — possible related causes or categories worth exploring, and any
     general next-step suggestions. This is a NUDGE for the doctor to consider,
     NOT a diagnosis and NOT a ranked differential. Do not assert likelihood
     or probability. Do not say a condition is likely, probable, or confirmed.
     Phrase everything as open possibilities (e.g. "may be worth considering",
     "could be related to", "the doctor may wish to evaluate for"). Keep this
     brief — 2-3 sentences. If the symptoms are too vague or general to suggest
     anything responsibly, write "No specific considerations suggested — insufficient detail."

3. ABSOLUTE RULE FOR FIELDS 1-4 — DO NOT HALLUCINATE: Every sentence in
   chief_complaint, duration, severity, and history must be traceable to
   something the patient actually said. "More descriptive" means fully
   expressing what they said in complete, natural clinical language — NOT
   adding new symptoms, causes, timelines, or details they never mentioned.
4. If a field truly was not mentioned at all, write "Not mentioned by patient"
   rather than guessing or filling in a plausible-sounding detail.
5. Fields 1-4 must not contain clinical interpretations, diagnoses, or
   assumptions the patient did not state — that reasoning belongs ONLY in
   possible_recommendations, and even there, must stay conservative and hedged.
6. Respond with ONLY valid JSON. No explanatory text before or after the JSON.

EXAMPLE:
Patient speech: "My stomach has been hurting me since yesterday, I've been vomiting too, it's very bad, I can't even eat anything"
Output:
{{
  "chief_complaint": "The patient reports abdominal pain accompanied by vomiting. The pain is significant enough that the patient has been unable to eat since symptoms began.",
  "duration": "Symptoms began yesterday and have persisted since onset. The patient did not specify whether the pain has changed in intensity over this period.",
  "severity": "The patient describes the pain as very bad, and it has been severe enough to prevent normal eating, indicating a severe presentation.",
  "history": "Not mentioned by patient.",
  "possible_recommendations": "Given the combination of abdominal pain and vomiting, general gastrointestinal causes may be worth considering, though this is not exhaustive. The doctor may wish to evaluate hydration status given the reported inability to eat."
}}

Now process this patient's speech:
Patient speech: {transcript}

Output (JSON only, no other text):"""


# ============================================================================
# LLM CALLING
# ============================================================================

def call_natlas_llm(prompt: str, max_new_tokens: int = 750) -> str:
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
    """Ensure all required fields exist; fill missing ones with defaults."""
    required_fields = ["chief_complaint", "duration", "severity", "history"]
    for field in required_fields:
        if field not in data or not data[field]:
            data[field] = "Not mentioned"
    if "possible_recommendations" not in data or not data["possible_recommendations"]:
        data["possible_recommendations"] = "No specific considerations suggested — insufficient detail."
    return data


def parse_llm_output(raw_output: str) -> dict:
    """Extract, parse, and validate the JSON clinical note from raw LLM text."""
    if not raw_output:
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "", "possible_recommendations": "",
                "_error": "Empty LLM output"}

    json_str = extract_json_from_text(raw_output)
    if not json_str:
        logger.warning("No JSON found in LLM output")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "", "possible_recommendations": "",
                "_error": "No JSON in output", "_raw": raw_output[:200]}

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        return validate_structure(data)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "", "possible_recommendations": "",
                "_error": f"JSON parse failed: {str(e)}", "_raw": json_str[:200]}


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structure_note(transcript: str, language_context: str = "") -> dict:
    """
    Main function: Convert raw patient transcript -> structured clinical note.

    Args:
        transcript: Patient's raw speech (Yoruba/Igbo/Hausa/English)
        language_context: Optional context string from the audio-based
            language detection stage (nlp/audio_language_detect.py), run
            BEFORE ASR on the raw audio. Passing this in means N-ATLaS
            doesn't have to figure out the language situation itself.

    Returns:
        Dict with chief_complaint, duration, severity, history,
        possible_recommendations (all English).
        May include _error / _raw keys if something went wrong.
    """
    if not transcript or len(transcript.strip()) == 0:
        logger.warning("Empty transcript provided")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "", "possible_recommendations": "",
                "_error": "Empty transcript"}

    logger.info(f"Structuring note from transcript: {transcript[:50]}...")

    prompt = STRUCTURE_PROMPT.format(
        transcript=transcript,
        language_context=language_context or "Language: not pre-detected — determine from the transcript itself."
    )

    try:
        raw_output = call_natlas_llm(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "", "possible_recommendations": "",
                "_error": f"LLM call failed: {str(e)}"}

    structured = parse_llm_output(raw_output)
    logger.info("Note structuring complete")
    return structured


def structure_note_batch(transcripts: list) -> list:
    """Process multiple transcripts (useful for testing)."""
    return [structure_note(t) for t in transcripts]
