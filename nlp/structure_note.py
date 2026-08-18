"""
Structuring stage — sends the raw transcript to N-ATLaS-LLM and parses
the response into a structured clinical note (dict/JSON).
"""
import json

PROMPT_TEMPLATE = """Extract the following fields as JSON from this patient
transcript. Fields: chief_complaint, duration, severity, history.
Do not invent information not present in the transcript.

Transcript: {transcript}

Respond with JSON only."""

def structure_note(transcript: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(transcript=transcript)
    # TODO: call local model or NATLAS_ENDPOINT_URL here, get raw text back
    raw_output = "{}"  # placeholder
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"chief_complaint": "", "duration": "", "severity": "", "history": "",
                "_raw": raw_output, "_parse_error": True}
