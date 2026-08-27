Lightweight Language & Code-Switching Detection

Runs BEFORE N-ATLaS, so the 8B model isn't asked to also figure out what
language(s) it's dealing with — it's just told directly. This is a cheap,
fast heuristic pass, not a full ML model, kept intentionally lightweight
to avoid adding meaningful latency of its own.

Output gets injected into the N-ATLaS prompt as context, e.g.:
  "Detected: primarily Hausa, with some English code-switching present."
"""

import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Small, fast reference word lists — not exhaustive, just enough to flag
# likely English content mixed into a non-English transcript.
COMMON_ENGLISH_WORDS = {
    "the", "and", "is", "are", "have", "has", "since", "very", "pain",
    "since", "feel", "feeling", "day", "days", "week", "weeks", "yesterday",
    "today", "morning", "night", "hospital", "doctor", "medicine", "tablet",
    "injection", "fever", "headache", "stomach", "body", "sick", "not",
    "bad", "much", "little", "small", "big", "please", "help", "I", "my",
}


def detect_code_switching(transcript: str, expected_language: str, threshold: float = 0.15) -> dict:
    """
    Lightweight heuristic: estimate whether the transcript is likely
    code-switched (mixing English into the expected local language).

    This does NOT attempt to identify language per-word or per-segment —
    it's a cheap ratio check, meant only to give N-ATLaS a heads-up, not
    to make a confident linguistic determination.

    Args:
        transcript: Raw ASR transcript text
        expected_language: The language selected/detected for ASR (Hausa/Igbo/Yoruba)
        threshold: Fraction of recognizable English words above which we
                   flag likely code-switching

    Returns:
        {
            "expected_language": str,
            "likely_code_switched": bool,
            "english_word_ratio": float,
            "context_note": str   # human-readable string to inject into the LLM prompt
        }
    """
    if not transcript or not transcript.strip():
        return {
            "expected_language": expected_language,
            "likely_code_switched": False,
            "english_word_ratio": 0.0,
            "context_note": f"Language: {expected_language}. No code-switching detected (empty transcript)."
        }

    words = re.findall(r"\b\w+\b", transcript.lower())
    if not words:
        ratio = 0.0
    else:
        english_hits = sum(1 for w in words if w in COMMON_ENGLISH_WORDS)
        ratio = english_hits / len(words)

    likely_code_switched = ratio > threshold

    if likely_code_switched:
        context_note = (
            f"Language: primarily {expected_language}, but this transcript appears to mix in "
            f"English words or phrases (code-switching detected, ~{ratio:.0%} recognizable English "
            f"terms). Translate the full meaning consistently into English regardless of which "
            f"language each part was spoken in."
        )
    else:
        context_note = f"Language: {expected_language}. No significant code-switching detected."

    logger.info(f"Code-switching check: {expected_language}, ratio={ratio:.2f}, flagged={likely_code_switched}")

    return {
        "expected_language": expected_language,
        "likely_code_switched": likely_code_switched,
        "english_word_ratio": round(ratio, 3),
        "context_note": context_note,
    }
