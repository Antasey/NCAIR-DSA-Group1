"""
LLM Structuring Stage — converts raw patient transcripts into structured 
English clinical notes using N-ATLaS LLM.

This is the CORE of NCAIR-DSA. It handles:
1. Translation (Yoruba/Igbo/Hausa -> English)
2. Formatting (raw speech -> structured JSON)
3. Validation (ensures required fields present)
4. Error handling (graceful fallback if LLM fails)

Deploy mode: Uses Hugging Face Inference Endpoint (no local GPU needed)
"""

import os
import json
import re
import requests
import logging
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
NATLAS_ENDPOINT_URL = os.getenv("NATLAS_ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

if not NATLAS_ENDPOINT_URL or not HF_TOKEN:
    logger.warning(
        "NATLAS_ENDPOINT_URL or HF_TOKEN not set. "
        "LLM calls will fail. Set in .env file."
    )

# ============================================================================
# PROMPT ENGINEERING
# ============================================================================

STRUCTURE_PROMPT = """You are a clinical note assistant specializing in multilingual healthcare.

Your task: Take a patient's raw speech (in Yoruba, Igbo, Hausa, or English) 
and structure it into a clean, organized English clinical note.

INSTRUCTIONS:
1. If the patient spoke in Yoruba, Igbo, or Hausa, translate their meaning into clear English.
2. Extract and organize EXACTLY these four fields:
   - chief_complaint: The patient's main symptom or concern (1-2 sentences, in English)
   - duration: How long the problem has lasted (e.g., "since yesterday", "3 days", "1 week")
   - severity: How bad it is on a scale (mild, moderate, or severe) — infer from patient's language
   - history: Any other relevant details or secondary symptoms they mentioned

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

# Optional: Keep alternate prompts for A/B testing
STRUCTURE_PROMPT_STRICT = """You are a clinical documentation assistant. 

TASK: Convert this patient's speech into structured JSON for clinical intake.
Patient's speech may be in Yoruba, Igbo, Hausa, or English. Translate to English.

OUTPUT FORMAT (valid JSON only):
{{
  "chief_complaint": "Main symptom/concern stated by patient",
  "duration": "Time period of problem",
  "severity": "mild|moderate|severe",
  "history": "Other relevant details mentioned"
}}

RULES:
- Extract ONLY what patient said. Do not invent.
- If not mentioned, use "Not mentioned"
- Translate all non-English speech to English
- Return ONLY the JSON object, nothing else

Patient: {transcript}

JSON:"""

# ============================================================================
# LLM CALLING
# ============================================================================

def call_natlas_llm(prompt: str, timeout: int = 30) -> str:
    """
    Call N-ATLaS LLM via Hugging Face Inference Endpoint.
    
    Args:
        prompt: The prompt to send to the model
        timeout: Request timeout in seconds (model may take time warming up)
    
    Returns:
        Raw model output (may need parsing)
    
    Raises:
        requests.RequestException: If the API call fails
    """
    if not NATLAS_ENDPOINT_URL or not HF_TOKEN:
        raise ValueError(
            "NATLAS_ENDPOINT_URL and HF_TOKEN must be set in .env file"
        )
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,  # Enough for structured note
            "temperature": 0.3,      # Low = consistent, structured output
            "top_p": 0.9,           # Nucleus sampling
            "do_sample": True
        }
    }
    
    logger.info("Calling N-ATLaS LLM...")
    
    try:
        response = requests.post(
            NATLAS_ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        # HF Inference API returns list of results
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return str(result)
    
    except requests.Timeout:
        logger.error("LLM request timed out")
        raise
    except requests.RequestException as e:
        logger.error(f"LLM API error: {e}")
        raise

# ============================================================================
# JSON PARSING & VALIDATION
# ============================================================================

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain other content.
    Handles markdown code fences and embedded JSON.
    """
    # Remove markdown code fences
    cleaned = text.replace("```json", "").replace("```", "").strip()
    
    # Look for JSON object pattern: { ... }
    # This regex handles nested braces reasonably well
    json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
    match = re.search(json_pattern, cleaned, re.DOTALL)
    
    if match:
        return match.group(0)
    
    # If no match found, assume entire cleaned text is JSON
    if cleaned.startswith("{"):
        return cleaned
    
    return None

def validate_structure(data: dict) -> dict:
    """
    Ensure all required fields exist in the parsed JSON.
    Fill in missing fields with "Not mentioned".
    """
    required_fields = ["chief_complaint", "duration", "severity", "history"]
    
    for field in required_fields:
        if field not in data or not data[field]:
            data[field] = "Not mentioned"
    
    return data

def parse_llm_output(raw_output: str) -> dict:
    """
    Parse LLM output, extract JSON, validate structure.
    
    Handles:
    - Markdown code fences: ` ```json {...} ``` `
    - Embedded JSON: "Here's the output: {...}"
    - Malformed JSON (returns error dict)
    - Missing fields (fills with "Not mentioned")
    
    Returns:
        Dict with chief_complaint, duration, severity, history
        If parsing fails, returns error dict with _error and _raw fields
    """
    if not raw_output:
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "Empty LLM output"
        }
    
    # Extract JSON from possibly surrounding text
    json_str = extract_json_from_text(raw_output)
    
    if not json_str:
        logger.warning("No JSON found in LLM output")
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "No JSON in output",
            "_raw": raw_output[:200]
        }
    
    # Parse JSON
    try:
        data = json.loads(json_str)
        
        # Ensure it's a dict
        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary")
        
        # Validate and fill missing fields
        data = validate_structure(data)
        
        logger.info("Successfully parsed and validated clinical note")
        return data
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": f"JSON parse failed: {str(e)}",
            "_raw": json_str[:200]
        }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def structure_note(transcript: str) -> dict:
    """
    Main function: Convert raw patient transcript -> structured clinical note.
    
    This is the pipeline:
    1. Take raw patient speech (any language)
    2. Build structured prompt for N-ATLaS
    3. Call LLM
    4. Parse JSON output
    5. Validate and return
    
    Args:
        transcript: Patient's raw speech (Yoruba/Igbo/Hausa/English)
    
    Returns:
        Dict with keys:
        - chief_complaint (str): Patient's main symptom in English
        - duration (str): How long they've had the problem
        - severity (str): mild/moderate/severe
        - history (str): Other relevant details
        - _error (str, optional): Error message if something went wrong
        - _raw (str, optional): Raw LLM output if parsing failed
    """
    
    # Validate input
    if not transcript or len(transcript.strip()) == 0:
        logger.warning("Empty transcript provided")
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "Empty transcript"
        }
    
    logger.info(f"Structuring note from transcript: {transcript[:50]}...")
    
    # Build prompt
    prompt = STRUCTURE_PROMPT.format(transcript=transcript)
    
    # Call LLM
    try:
        raw_output = call_natlas_llm(prompt)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": f"LLM call failed: {str(e)}"
        }
    
    # Parse output
    structured = parse_llm_output(raw_output)
    
    logger.info("Note structuring complete")
    return structured

# ============================================================================
# BATCH PROCESSING (for testing)
# ============================================================================

def structure_note_batch(transcripts: list) -> list:
    """
    Process multiple transcripts. Useful for testing or batch operations.
    
    Args:
        transcripts: List of transcript strings
    
    Returns:
        List of structured note dicts
    """
    results = []
    for i, transcript in enumerate(transcripts):
        logger.info(f"Processing transcript {i+1}/{len(transcripts)}")
        result = structure_note(transcript)
        results.append(result)
    return results

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test cases to validate the module works
    test_transcripts = [
        "My stomach has been hurting me since yesterday, I've been vomiting, it's very bad",
        "Headache for three days, mild, took medicine but didn't help much",
        "I have a cough that started last week, getting worse, no fever"
    ]
    
    print("Testing structure_note() function...\n")
    
    for transcript in test_transcripts:
        print(f"Input: {transcript}")
        result = structure_note(transcript)
        print(f"Output: {json.dumps(result, indent=2)}\n")
        print("-" * 60 + "\n")
