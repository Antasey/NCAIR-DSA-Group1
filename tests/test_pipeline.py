"""
Tests for NCAIR-DSA pipeline modules.

Covers what CAN be tested without a GPU or live model calls:
- Keyword extraction (pure logic, no model needed)
- LLM output parsing/validation (tests the parsing code, not the model itself)
- Database save/retrieve (SQLite, no external dependencies)

What's intentionally NOT covered here (needs a live model, run manually):
- transcribe_audio() — needs a real ASR model + real audio file
- structure_note() — needs the real N-ATLaS model loaded

Run with: pytest tests/test_pipeline.py -v
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
import pytest

# Make repo root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nlp.extract_keywords import extract_keywords
from nlp.structure_note import extract_json_from_text, validate_structure, parse_llm_output


# ============================================================================
# KEYWORD EXTRACTION TESTS
# ============================================================================

def test_extract_keywords_finds_symptoms():
    text = "Patient reports abdominal pain and vomiting"
    result = extract_keywords(text)
    assert "pain" in result["symptoms"] or "abdominal pain" in result["symptoms"]
    assert "vomiting" in result["symptoms"]


def test_extract_keywords_finds_severity():
    text = "The pain is severe and unbearable"
    result = extract_keywords(text)
    assert "severe" in result["severity"]


def test_extract_keywords_finds_duration():
    text = "Symptoms started 3 days ago"
    result = extract_keywords(text)
    assert any("3" in d for d in result["duration"])


def test_extract_keywords_finds_anatomical_sites():
    text = "Pain in the chest and stomach"
    result = extract_keywords(text)
    assert "chest" in result["anatomical_sites"]
    assert "stomach" in result["anatomical_sites"]


def test_extract_keywords_empty_text_returns_empty_lists():
    result = extract_keywords("")
    assert result["symptoms"] == []
    assert result["duration"] == []
    assert result["severity"] == []
    assert result["anatomical_sites"] == []


def test_extract_keywords_no_matches_returns_empty_lists():
    text = "The weather is nice today"
    result = extract_keywords(text)
    assert result["symptoms"] == []


def test_extract_keywords_case_insensitive():
    text = "PATIENT HAS SEVERE FEVER"
    result = extract_keywords(text)
    assert "fever" in result["symptoms"]
    assert "severe" in result["severity"]


# ============================================================================
# LLM OUTPUT PARSING TESTS
# (These test the parsing logic, not the model — no GPU needed)
# ============================================================================

def test_extract_json_from_markdown_fence():
    raw = '''```json
{
  "chief_complaint": "Headache",
  "duration": "2 days",
  "severity": "moderate",
  "history": "None"
}
```'''
    result = extract_json_from_text(raw)
    assert result is not None
    assert '"chief_complaint"' in result


def test_extract_json_from_embedded_text():
    raw = 'Here is the note: {"chief_complaint": "Fever", "duration": "1 week", "severity": "severe", "history": "Flu-like"}'
    result = extract_json_from_text(raw)
    assert result is not None
    parsed = json.loads(result)
    assert parsed["chief_complaint"] == "Fever"


def test_extract_json_returns_none_for_no_json():
    raw = "I cannot process this request."
    result = extract_json_from_text(raw)
    assert result is None


def test_validate_structure_fills_missing_fields():
    incomplete = {"chief_complaint": "Pain"}
    result = validate_structure(incomplete)
    assert result["duration"] == "Not mentioned"
    assert result["severity"] == "Not mentioned"
    assert result["history"] == "Not mentioned"
    assert result["chief_complaint"] == "Pain"  # unchanged


def test_validate_structure_keeps_existing_fields():
    complete = {
        "chief_complaint": "Headache",
        "duration": "2 days",
        "severity": "mild",
        "history": "None"
    }
    result = validate_structure(complete)
    assert result == complete


def test_parse_llm_output_valid_json():
    raw = '{"chief_complaint": "Abdominal pain", "duration": "Since yesterday", "severity": "severe", "history": "Nausea present"}'
    result = parse_llm_output(raw)
    assert result["chief_complaint"] == "Abdominal pain"
    assert result["severity"] == "severe"
    assert "_error" not in result


def test_parse_llm_output_malformed_json_returns_error():
    raw = "{ this is not valid json }"
    result = parse_llm_output(raw)
    assert "_error" in result
    assert result["chief_complaint"] == ""


def test_parse_llm_output_empty_string_returns_error():
    result = parse_llm_output("")
    assert "_error" in result


def test_parse_llm_output_markdown_wrapped_json():
    raw = '''```json
{"chief_complaint": "Cough", "duration": "1 week", "severity": "mild", "history": "No fever"}
```'''
    result = parse_llm_output(raw)
    assert result["chief_complaint"] == "Cough"
    assert "_error" not in result


# ============================================================================
# DATABASE TESTS
# ============================================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Creates a fresh temporary database for each test, so tests don't
    interfere with each other or with the real notes.db."""
    import templates.clinical_note as clinical_note

    test_db_path = tmp_path / "test_notes.db"
    monkeypatch.setattr(clinical_note, "DB_PATH", test_db_path)
    clinical_note.init_db()
    return clinical_note


def test_init_db_creates_tables(temp_db):
    conn = sqlite3.connect(temp_db.DB_PATH)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "patients" in tables
    assert "clinical_visits" in tables


def test_save_and_retrieve_patient_record(temp_db):
    temp_db.save_patient_record(
        patient_id="P001",
        chief_complaint="Headache",
        duration="2 days",
        severity="mild",
        history="None",
        language="Hausa",
        keywords={"symptoms": ["headache"]},
        transcript="raw transcript text"
    )

    history = temp_db.get_patient_history("P001")
    assert len(history) == 1
    assert history[0][1] == "Headache"  # chief_complaint column


def test_multiple_visits_same_patient(temp_db):
    temp_db.save_patient_record(
        patient_id="P002", chief_complaint="Fever", duration="1 day",
        severity="mild", history="None", language="Igbo",
        keywords={}, transcript="visit 1"
    )
    temp_db.save_patient_record(
        patient_id="P002", chief_complaint="Cough", duration="3 days",
        severity="moderate", history="None", language="Igbo",
        keywords={}, transcript="visit 2"
    )

    history = temp_db.get_patient_history("P002")
    assert len(history) == 2


def test_get_history_unknown_patient_returns_empty(temp_db):
    history = temp_db.get_patient_history("NONEXISTENT")
    assert history == []


def test_visits_ordered_most_recent_first(temp_db):
    import time
    temp_db.save_patient_record(
        patient_id="P003", chief_complaint="First visit", duration="", 
        severity="", history="", language="Yoruba", keywords={}, transcript=""
    )
    time.sleep(1.1)  # ensure distinct timestamps
    temp_db.save_patient_record(
        patient_id="P003", chief_complaint="Second visit", duration="",
        severity="", history="", language="Yoruba", keywords={}, transcript=""
    )

    history = temp_db.get_patient_history("P003")
    assert history[0][1] == "Second visit"  # most recent first
    assert history[1][1] == "First visit"


# ============================================================================
# INTEGRATION-STYLE TEST (still no live model — uses mock note data)
# ============================================================================

def test_keyword_extraction_runs_on_english_note_not_raw_transcript():
    """
    Regression test for the bug where keywords were extracted from the raw
    (non-English) transcript instead of the translated English note.
    Keywords should match against English clinical text.
    """
    english_note = "Chief complaint: severe abdominal pain since yesterday with vomiting"
    result = extract_keywords(english_note)
    assert len(result["symptoms"]) > 0
    assert "severe" in result["severity"]
