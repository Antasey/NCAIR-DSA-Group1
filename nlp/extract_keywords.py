"""
Keyword extraction from patient speech — identifies clinical entities
(symptoms, duration, severity, anatomical mentions) to highlight in the
structured note for doctor review.

Uses the SAME N-ATLaS transformers model as structure_note.py (shared
instance, so no second model copy is loaded in RAM).

The EXTRACTION itself runs on the LLM — the only regex here is for
highlighting keywords in HTML, not for extracting them.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Final

from structure_note import generate_text

logger = logging.getLogger(__name__)

EXTRACT_PROMPT: Final = """You are reviewing a patient transcript for a triage highlight view.

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

_EMPTY_RESULT: Final = {"symptoms": [], "duration": [], "severity": [], "anatomical_sites": []}


def extract_keywords(transcript: str) -> dict[str, list[str]]:
    """
    Extract clinical keywords from patient transcript via N-ATLaS LLM.
    Returns dict with categories: symptoms, duration, severity, anatomical_sites.

    NEVER returns an empty dict silently — if the LLM fails or returns
    unparseable output, we log the issue and return empty lists with a
    visible marker so the app can show "No keywords extracted" instead
    of a blank box.
    """
    if not transcript or not transcript.strip():
        return {k: [] for k in _EMPTY_RESULT}

    try:
        raw = generate_text(
            EXTRACT_PROMPT.format(transcript=transcript.strip()),
            max_new_tokens=250,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning("Keyword extraction: LLM generation failed: %s", e)
        return {k: [] for k in _EMPTY_RESULT}

    # Clean markdown fences
    cleaned = raw.replace("```json", "").replace("```", "").strip()

    # Extract JSON block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        logger.warning("Keyword extraction: no JSON found in model output: %s", raw[:200])
        return {k: [] for k in _EMPTY_RESULT}

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning("Keyword extraction: JSON parse failed: %s", e)
        return {k: [] for k in _EMPTY_RESULT}

    if not isinstance(data, dict):
        logger.warning("Keyword extraction: parsed data is not a dict")
        return {k: [] for k in _EMPTY_RESULT}

    extracted: dict[str, list[str]] = {}
    for key in _EMPTY_RESULT:
        values = data.get(key, [])
        if not isinstance(values, list):
            values = []
        # Deduplicate, strip whitespace, filter empty, limit to 20 per category
        seen = set()
        deduped = []
        for v in values:
            s = str(v).strip()
            if s and s.lower() not in seen and len(deduped) < 20:
                seen.add(s.lower())
                deduped.append(s)
        extracted[key] = deduped

    return extracted


def _highlight_text(text: str, keywords: list[str]) -> str:
    """Highlight keywords in text with yellow background using regex."""
    if not text or not keywords:
        return text

    # Sort by length (longest first) to avoid partial replacements
    unique_keywords = sorted(set(keywords), key=len, reverse=True)

    highlighted = text
    for keyword in unique_keywords:
        if not keyword:
            continue
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f'<mark style="background-color: #FFFF00;">{m.group(0)}</mark>',
            highlighted,
        )
    return highlighted


def highlight_keywords(note: dict, keywords: dict[str, list[str]]) -> str:
    """
    Render the structured note as HTML with keywords highlighted.
    Doctor sees the note with color-coded keywords for quick scanning.
    """
    all_keywords = []
    for items in keywords.values():
        all_keywords.extend(items)

    html = "<div style='font-family: Arial; line-height: 1.6;'>"

    for field, label in [
        ("chief_complaint", "Chief Complaint"),
        ("duration", "Duration"),
        ("severity", "Severity"),
        ("history", "Relevant History"),
    ]:
        value = note.get(field, "")
        html += f"<h4>{label}</h4>"
        html += f"<p>{_highlight_text(value, all_keywords)}</p>"

    html += "</div>"
    return html
