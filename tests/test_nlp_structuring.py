"""
Tests for LLM Structuring Module (nlp/structure_note.py)

Run with: pytest tests/test_nlp_structuring.py -v
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nlp.structure_note import (
    _extract_json_from_text,
    _validate_structure,
    _parse_llm_output,
    structure_note,
    _error_result,
)


# ============================================================================
# PARSING TESTS (no model needed)
# ============================================================================

class TestJSONParsing:
    def test_parse_clean_json(self):
        raw = '{"chief_complaint": "Headache", "duration": "2 days", "severity": "Moderate", "history": "No previous episodes", "possible_recommendations": "Rest"}'
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Headache"
        assert result["duration"] == "2 days"
        assert "_error" not in result

    def test_parse_json_with_preamble(self):
        raw = """You are a clinical assistant. Here's the structured note:
        {
            "chief_complaint": "Abdominal pain",
            "duration": "Since yesterday",
            "severity": "Severe",
            "history": "Patient reports vomiting",
            "possible_recommendations": "Check hydration"
        }
        Let me know if you need anything else."""
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Abdominal pain"
        assert "_error" not in result

    def test_parse_malformed_json(self):
        raw = '{"chief_complaint": Headache, "duration": "2 days"}'
        result = _parse_llm_output(raw)
        assert "_error" in result
        assert result["chief_complaint"] == ""

    def test_parse_no_json(self):
        raw = "I couldn't parse that."
        result = _parse_llm_output(raw)
        assert "_error" in result
        assert "No JSON" in result["_error"]

    def test_parse_missing_fields(self):
        raw = '{"chief_complaint": "Fever", "severity": "Mild"}'
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Fever"
        assert result["severity"] == "Mild"
        assert result["duration"] == "Not mentioned by patient"
        assert result["history"] == "Not mentioned by patient"

    def test_parse_nested_braces_in_values(self):
        raw = '{"chief_complaint": "Pain (located {here})", "duration": "2 days", "severity": "mild", "history": "None", "possible_recommendations": "N/A"}'
        result = _parse_llm_output(raw)
        assert result["chief_complaint"] == "Pain (located {here})"
        assert "_error" not in result

    def test_extract_json_from_nested_text(self):
        raw = 'prefix {"a": 1} suffix {"b": 2}'
        result = _extract_json_from_text(raw)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == {"a": 1}


class TestValidateStructure:
    def test_fills_all_missing_fields(self):
        data = {"chief_complaint": "Pain"}
        result = _validate_structure(data)
        assert result["duration"] == "Not mentioned by patient"
        assert result["severity"] == "Not mentioned by patient"
        assert result["history"] == "Not mentioned by patient"
        assert result["possible_recommendations"] == "No specific considerations suggested — insufficient detail."

    def test_preserves_existing_fields(self):
        data = {
            "chief_complaint": "Headache",
            "duration": "2 days",
            "severity": "mild",
            "history": "None",
            "possible_recommendations": "Rest",
        }
        result = _validate_structure(data)
        assert result == data


class TestErrorResult:
    def test_error_result_structure(self):
        result = _error_result("Something broke", "raw text")
        assert result["_error"] == "Something broke"
        assert result["_raw"] == "raw text"
        assert all(result.get(k) == "" for k in ["chief_complaint", "duration", "severity", "history", "possible_recommendations"])


# ============================================================================
# INTEGRATION TESTS (mocked model)
# ============================================================================

class TestStructureNoteWithMock:
    def test_structure_note_success(self, monkeypatch):
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            return json.dumps({
                "chief_complaint": "Headache",
                "duration": "2 days",
                "severity": "moderate",
                "history": "None",
                "possible_recommendations": "Rest and hydration",
            })
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        result = structure_note("I have a headache for 2 days")
        assert "_error" not in result
        assert result["chief_complaint"] == "Headache"

    def test_structure_note_empty_transcript(self):
        result = structure_note("")
        assert "_error" in result
        assert "Empty transcript" in result["_error"]

    def test_structure_note_llm_failure_with_retry(self, monkeypatch):
        calls = []
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            calls.append(1)
            if len(calls) == 1:
                return "not json"
            return json.dumps({
                "chief_complaint": "Pain",
                "duration": "1 day",
                "severity": "severe",
                "history": "None",
                "possible_recommendations": "See doctor",
            })
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        result = structure_note("My head hurts")
        assert "_error" not in result
        assert result["chief_complaint"] == "Pain"
        assert len(calls) == 2

    def test_structure_note_all_retries_fail(self, monkeypatch):
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            return "garbage"
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        result = structure_note("Some symptoms")
        assert "_error" in result
        assert "after 2 attempts" in result["_error"]

    def test_structure_note_llm_exception(self, monkeypatch):
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            raise RuntimeError("GPU OOM")
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        result = structure_note("Some symptoms")
        assert "_error" in result
        assert "GPU OOM" in result["_error"]

    def test_structure_note_empty_fields_guard(self, monkeypatch):
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            return json.dumps({
                "chief_complaint": "",
                "duration": "",
                "severity": "",
                "history": "",
                "possible_recommendations": "",
            })
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        result = structure_note("Some symptoms")
        assert "_error" in result
        assert "empty" in result["_error"].lower()


# ============================================================================
# BATCH PROCESSING TEST
# ============================================================================

class TestBatchProcessing:
    def test_batch_processes_multiple(self, monkeypatch):
        def mock_generate(prompt, max_new_tokens=750, temperature=0.3):
            return json.dumps({
                "chief_complaint": "Pain",
                "duration": "1 day",
                "severity": "mild",
                "history": "None",
                "possible_recommendations": "Rest",
            })
        monkeypatch.setattr("nlp.structure_note.generate_text", mock_generate)

        from nlp.structure_note import structure_note_batch
        results = structure_note_batch([
            ("transcript 1", "Hausa"),
            ("transcript 2", "Yoruba"),
        ])
        assert len(results) == 2
        assert all("_error" not in r for r in results)
