"""
Keyword extraction from patient speech — identifies clinical entities
(symptoms, duration, severity, anatomical mentions) to highlight in the
structured note for doctor review.

Uses the locally loaded N-ATLaS model (shared with structure_note.py,
so no second model copy is loaded) instead of a fixed keyword list —
regex/word-lists miss anything not anticipated in advance. Nothing
here is saved; it's purely for the triage highlight view.
"""
import json
import logging
import re
from typing import List, Dict

from structure_note import get_llm

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = """You are reviewing a patient transcript for a triage highlight view.

Identify clinically relevant terms actually present in the transcript below,
sorted into these four categories:
- symptoms: what the patient is experiencing
- duration: any timing/frequency phrases (e.g. "3 days", "since yesterday")
- severity: any severity/intensity language the patient used
- anatomical_sites: any body parts/areas mentioned

Only include terms that literally appear in or are directly stated by the
transcript — do not infer anything not said. If a category has nothing,
return an empty list for it. Respond with ONLY valid JSON, no other text:
{{"symptoms": [], "duration": [], "severity": [], "anatomical_sites": []}}

Transcript: {transcript}

Output (JSON only):"""

_EMPTY_RESULT = {"symptoms": [], "duration": [], "severity": [], "anatomical_sites": []}


def extract_keywords(transcript: str) -> Dict[str, List[str]]:
    """
    Extract clinical keywords from patient transcript via N-ATLaS.
    Returns dict with categories: symptoms, duration, severity, anatomical_sites.
    Falls back to empty lists (never raises) if the model output isn't
    parseable, so a bad generation never breaks the highlight view.
    """
    if not transcript or not transcript.strip():
        return {k: [] for k in _EMPTY_RESULT}

    llm = get_llm()  # reuses the same in-memory model as structure_note.py

    result = llm.create_completion(
        EXTRACT_PROMPT.format(transcript=transcript),
        max_tokens=250,
        temperature=0.1,
        top_p=0.9,
    )
    raw = result["choices"][0]["text"].strip()

    cleaned = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("Keyword extraction: no JSON found in model output")
        return {k: [] for k in _EMPTY_RESULT}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"Keyword extraction: JSON parse failed: {e}")
        return {k: [] for k in _EMPTY_RESULT}

    extracted = {}
    for key in _EMPTY_RESULT:
        values = data.get(key, [])
        if not isinstance(values, list):
            values = []
        extracted[key] = list(dict.fromkeys(str(v).strip() for v in values if str(v).strip()))

    return extracted

def highlight_keywords(note: Dict, keywords: Dict[str, List[str]]) -> str:
    """
    Render the structured note as HTML with keywords highlighted.
    Doctor sees the note with color-coded keywords for quick scanning.
    """
    all_keywords = []
    for category, items in keywords.items():
        all_keywords.extend(items)
    
    html = "<div style='font-family: Arial; line-height: 1.6;'>"
    
    # Chief Complaint
    chief = note.get("chief_complaint", "")
    html += f"<h4>Chief Complaint</h4>"
    html += f"<p>{_highlight_text(chief, all_keywords)}</p>"
    
    # Duration
    duration = note.get("duration", "")
    html += f"<h4>Duration</h4>"
    html += f"<p>{_highlight_text(duration, keywords.get('duration', []))}</p>"
    
    # Severity
    severity = note.get("severity", "")
    html += f"<h4>Severity</h4>"
    html += f"<p>{_highlight_text(severity, keywords.get('severity', []))}</p>"
    
    # History
    history = note.get("history", "")
    html += f"<h4>Relevant History</h4>"
    html += f"<p>{_highlight_text(history, all_keywords)}</p>"
    
    html += "</div>"
    return html

def _highlight_text(text: str, keywords: List[str]) -> str:
    """
    Internal: highlight keywords in text with yellow background.
    """
    if not text:
        return text
    
    highlighted = text
    for keyword in set(keywords):  # avoid duplicate highlighting
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            f'<mark style="background-color: #FFFF00;">{keyword}</mark>',
            highlighted
        )
    return highlighted
