"""
Tests for NCAIR-DSA pipeline modules.

Covers what CAN be tested without a GPU or live model calls:
- Keyword extraction (mocked LLM)
- LLM output parsing/validation
- Database save/retrieve (SQLite, no external dependencies)

Run with: pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from nlp.structure_note import _extract_json_from_text, _validate_structure, _parse_llm_output


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    import templates.clinical_note as clinical_note
    test_db_path = tmp_path / "test_notes.db"
    monkeypatch.setattr(clinical_note, "DB_PATH", test_db_path)
    clinical_note.init_db()
    return clinical_note


@pytest.fixture
def mock_generate():
    """Mock generate_text to return predictable JSON."""
    def _generate(prompt: str, max_new_tokens: int = 750, temperature: float = 0.3):
        if "triage highlight" in prompt.lower():
            return json.dumps({
                "symptoms": ["fever", "headache"],
                "duration": ["2 days"],
                "severity": ["severe"],
                "anatomical_sites": ["head"],
            })
        return json.dumps({
            "chief_complaint": "Headache",
            "duration": "2 days",
            "severity": "moderate",
            "history": "None",
            "possible_recommendations": "Rest",
        })
    return _generate


# ============================================================================
# KEYWORD EXTRACTION TESTS (mocked LLM)
# ============================================================================

class TestExtractKeywords:
    def test_extract_keywords_basic(self, mock_generate):
        with patch("nlp.extract_keywords.generate_text", mock_generate):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords("Patient has fever and headache for 2 days")

        assert "fever" in result["symptoms"]
        assert "headache" in result["symptoms"]
        assert "2 days" in result["duration"]
        assert "severe" in result["severity"]
        assert "head" in result["anatomical_sites"]

    def test_extract_keywords_empty_text(self):
        with patch("nlp.extract_keywords.generate_text", return_value=""):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords("")
        assert result == {"symptoms": [], "duration": [], "severity": [], "anatomical_sites": []}

    def test_extract_keywords_no_json_fallback(self):
        with patch("nlp.extract_keywords.generate_text", return_value="not json"):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords("some text")
        assert all(v == [] for v in result.values())

    def test_extract_keywords_llm_unavailable(self):
        with patch("nlp.extract_keywords.generate_text", side_effect=RuntimeError("no model")):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords("some text")
        assert all(v == [] for v in result.values())

    def test_extract_keywords_deduplicates(self):
        def mock_gen(prompt, **kwargs):
            return json.dumps({
                "symptoms": ["fever", "Fever", "fever", "headache"],
                "duration": [], "severity": [], "anatomical_sites": [],
            })
        with patch("nlp.extract_keywords.generate_text", mock_gen):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords("text")
        assert result["symptoms"] == ["fever", "headache"]


# ============================================================================
# LLM OUTPUT PARSING TESTS
# ============================================================================

class TestJSONParsing:
    def test_extract_json_from_markdown_fence(self):
        raw = """```json
{"chief_complaint": "Headache", "duration": "2 days", "severity": "moderate", "history": "None"}
```"""
        result = _extract_json_from_text(raw)
        assert result is not None
        assert '"chief_complaint"' in result

    def test_extract_json_from_embedded_text(self):
        raw = 'Here is the note: {"chief_complaint": "Fever", "duration": "1 week", "severity": "severe", "history": "Flu-like"}'
        result = _extract_json_from_text(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["chief_complaint"] == "Fever"

    def test_extract_json_returns_none_for_no_json(self):
        raw = "I cannot process this request."
        result = _extract_json_from_text(raw)
        assert result is None

    def test_extract_json_handles_nested_braces(self):
        raw = '{"chief_complaint": "Pain (located {here})", "duration": "2 days", "severity": "mild", "history": "None"}'
        result = _extract_json_from_text(raw)
        assert result is not None
        parsed = json.loads(result)
        assert "Pain (located {here})" == parsed["chief_complaint"]

    def test_validate_structure_fills_missing_fields(self):
        incomplete = {"chief_complaint": "Pain"}
        result = _validate_structure(incomplete)
        assert result["duration"] == "Not mentioned by patient"
        assert result["severity"] == "Not mentioned by patient"
        assert result["history"] == "Not mentioned by patient"
        assert result["chief_complaint"] == "Pain"

    def test_validate_structure_keeps_existing_fields(self):
        complete = {
            "chief_complaint": "Headache",
            "duration": "2 days",
            "severity": "mild",
            "history": "None",
            "possible_recommendations": "Rest",
        }
        result = _validate_structure(complete)
        assert result == complete

    def test_parse_llm_output_valid_json(self):
        raw = '{"chief_complaint": "Abdominal pain", "duration": "Since yesterday", "severity": "severe", "history": "Nausea present"}'
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Abdominal pain"
        assert "_error" not in result

    def test_parse_llm_output_malformed_json_returns_error(self):
        raw = "{ this is not valid json }"
        result = _parse_llm_output(raw)
        assert "_error" in result
        assert result["chief_complaint"] == ""

    def test_parse_llm_output_empty_string_returns_error(self):
        result = _parse_llm_output("")
        assert "_error" in result

    def test_parse_llm_output_markdown_wrapped_json(self):
        raw = """```json
{"chief_complaint": "Cough", "duration": "1 week", "severity": "mild", "history": "No fever"}
```"""
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Cough"
        assert "_error" not in result


# ============================================================================
# DATABASE TESTS
# ============================================================================

class TestDatabase:
    def test_init_db_creates_tables(self, temp_db):
        with temp_db._get_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
        assert "patients" in tables
        assert "clinical_visits" in tables

    def test_save_and_retrieve_patient_record(self, temp_db):
        visit_id = temp_db.save_patient_record(
            patient_id="P001", chief_complaint="Headache", duration="2 days",
            severity="mild", history="None", possible_recommendations="Rest",
            language="Hausa", keywords={"symptoms": ["headache"]},
            transcript="raw transcript text",
        )
        assert isinstance(visit_id, int)
        history = temp_db.get_patient_history("P001")
        assert len(history) == 1
        assert history[0]["chief_complaint"] == "Headache"
        assert history[0]["status"] == temp_db.STATUS_PENDING

    def test_multiple_visits_same_patient(self, temp_db):
        temp_db.save_patient_record(
            patient_id="P002", chief_complaint="Fever", duration="1 day",
            severity="mild", history="None", possible_recommendations="",
            language="Igbo", keywords={}, transcript="visit 1",
        )
        temp_db.save_patient_record(
            patient_id="P002", chief_complaint="Cough", duration="3 days",
            severity="moderate", history="None", possible_recommendations="",
            language="Igbo", keywords={}, transcript="visit 2",
        )
        history = temp_db.get_patient_history("P002")
        assert len(history) == 2

    def test_get_history_unknown_patient_returns_empty(self, temp_db):
        history = temp_db.get_patient_history("NONEXISTENT")
        assert history == []

    def test_visits_ordered_most_recent_first(self, temp_db):
        temp_db.save_patient_record(
            patient_id="P003", chief_complaint="First visit", duration="",
            severity="", history="", possible_recommendations="",
            language="Yoruba", keywords={}, transcript="",
        )
        temp_db.save_patient_record(
            patient_id="P003", chief_complaint="Second visit", duration="",
            severity="", history="", possible_recommendations="",
            language="Yoruba", keywords={}, transcript="",
        )
        history = temp_db.get_patient_history("P003")
        assert history[0]["chief_complaint"] == "Second visit"
        assert history[1]["chief_complaint"] == "First visit"

    def test_update_visit_and_mark_reviewed(self, temp_db):
        visit_id = temp_db.save_patient_record(
            patient_id="P004", chief_complaint="Pain", duration="1 day",
            severity="severe", history="None", possible_recommendations="",
            language="Hausa", keywords={}, transcript="",
        )
        temp_db.update_visit_and_mark_reviewed(
            visit_id, "Updated Pain", "2 days", "moderate", "Some history", "Consider X-ray",
        )
        visit = temp_db.get_visit_by_id(visit_id)
        assert visit["chief_complaint"] == "Updated Pain"
        assert visit["status"] == temp_db.STATUS_REVIEWED
        assert visit["reviewed_at"] is not None

    def test_get_dashboard_stats(self, temp_db):
        temp_db.save_patient_record(
            patient_id="P005", chief_complaint="A", duration="", severity="",
            history="", possible_recommendations="", language="Hausa", keywords={}, transcript="",
        )
        temp_db.save_patient_record(
            patient_id="P006", chief_complaint="B", duration="", severity="",
            history="", possible_recommendations="", language="Igbo", keywords={}, transcript="",
        )
        stats = temp_db.get_dashboard_stats()
        assert stats["total_patients"] == 2
        assert stats["total_visits"] == 2
        assert stats["pending_review"] == 2

    def test_get_pending_visits_oldest_first(self, temp_db):
        temp_db.save_patient_record(
            patient_id="P007", chief_complaint="Old", duration="", severity="",
            history="", possible_recommendations="", language="Hausa", keywords={}, transcript="",
        )
        temp_db.save_patient_record(
            patient_id="P008", chief_complaint="New", duration="", severity="",
            history="", possible_recommendations="", language="Hausa", keywords={}, transcript="",
        )
        pending = temp_db.get_pending_visits()
        assert len(pending) == 2
        assert pending[0]["patient_id"] == "P007"
        assert pending[1]["patient_id"] == "P008"

    def test_export_single_visit(self, temp_db, tmp_path):
        visit_id = temp_db.save_patient_record(
            patient_id="P009", chief_complaint="Export test", duration="",
            severity="", history="", possible_recommendations="",
            language="Yoruba", keywords={}, transcript="raw",
        )
        path = tmp_path / "export.csv"
        result = temp_db.export_single_visit_to_csv(visit_id, str(path))
        assert Path(result).exists()

    def test_export_all_records(self, temp_db, tmp_path):
        temp_db.save_patient_record(
            patient_id="P010", chief_complaint="A", duration="", severity="",
            history="", possible_recommendations="", language="Hausa", keywords={}, transcript="",
        )
        path = tmp_path / "all.csv"
        result, count = temp_db.export_all_records_to_csv(str(path))
        assert Path(result).exists()
        assert count == 1


# ============================================================================
# INTEGRATION-STYLE TEST
# ============================================================================

class TestKeywordExtractionOnEnglishNote:
    def test_keyword_extraction_runs_on_english_note(self, mock_generate):
        english_note = "Chief complaint: severe abdominal pain since yesterday with vomiting"
        with patch("nlp.extract_keywords.generate_text", mock_generate):
            from nlp.extract_keywords import extract_keywords
            result = extract_keywords(english_note)
        assert len(result["symptoms"]) > 0
        assert "severe" in result["severity"]
