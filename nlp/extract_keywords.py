"""
extract.py

Secondary, reference-only keyword/entity extraction.

Instead of matching the note against a fixed vocabulary, N-ATLaS itself
identifies the clinically relevant terms — so it can surface things
that weren't anticipated ahead of time. This stays a sidebar, never
authoritative over the structured note from structure.py.
"""

import json
import logging
from typing import List

from structure import get_llm

logger = logging.getLogger(__name__)

EXTRACT_SYSTEM_PROMPT = (
    "You are reviewing a clinical intake note. Identify the clinically "
    "relevant keywords in it: symptoms, affected body parts/systems, "
    "duration/frequency terms, and anything else a clinician would want "
    "flagged at a glance. Only include terms that actually appear in or "
    "are directly stated by the note — do not infer symptoms that were "
    "not mentioned, and do not suggest diagnoses or causes. Respond with "
    "ONLY a JSON array of short strings, no commentary, for example: "
    '["fever", "3 days duration", "abdominal pain"].'
)


def extract_keywords(note_text: str) -> List[str]:
    """Ask the model to freely identify relevant keywords in a note.

    This is intentionally open-ended: the model isn't given a fixed
    list to search for, so it can surface terms it wasn't told to look
    for in advance. Reference/sidebar use only.
    """
    llm = get_llm()  # reuses the same in-memory model as structure.py

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": note_text},
        ],
        temperature=0.1,
        max_tokens=200,
    )

    raw = result["choices"][0]["message"]["content"].strip()

    try:
        keywords = json.loads(raw)
        if not isinstance(keywords, list):
            raise ValueError("expected a JSON array")
        keywords = [str(k).strip() for k in keywords if str(k).strip()]
    except (json.JSONDecodeError, ValueError):
        logger.warning("Keyword extraction did not return a valid JSON array")
        keywords = []

    return keywords
