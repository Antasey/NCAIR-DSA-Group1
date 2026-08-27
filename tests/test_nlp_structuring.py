"""
Tests for LLM Structuring Module (nlp/structure_note.py)

This file contains unit tests and integration tests for your module.
Run with: pytest tests/test_nlp_structuring.py -v

These tests help you catch:
- JSON parsing errors
- Missing fields
- Hallucinated content
- Code-switching issues
- Malformed output
"""

import pytest
import json
from nlp.structure_note import structure_note, _parse_model_output

# ============================================================================
# PARSING TESTS (these don't need the model loaded)
# ============================================================================

class TestJSONParsing:
    """Test the JSON extraction and parsing logic."""
    
    def test_parse_clean_json(self):
        """Test parsing well-formed JSON."""
        raw_output = '''{"chief_complaint": "Headache", "duration": "2 days", "severity": "Moderate", "history": "No previous episodes"}'''
        result = _parse_model_output(raw_output)
        
        assert result["chief_complaint"] == "Headache"
        assert result["duration"] == "2 days"
        assert result["severity"] == "Moderate"
        assert result["history"] == "No previous episodes"
        assert "_error" not in result
    
    def test_parse_json_with_preamble(self):
        """Test parsing JSON that has model preamble before it."""
        raw_output = '''You are a clinical assistant. Here's the structured note:
        {
            "chief_complaint": "Abdominal pain",
            "duration": "Since yesterday",
            "severity": "Severe",
            "history": "Patient reports vomiting"
        }
        Let me know if you need anything else.'''
        
        result = _parse_model_output(raw_output)
        assert result["chief_complaint"] == "Abdominal pain"
        assert "_error" not in result
    
    def test_parse_malformed_json(self):
        """Test handling of malformed JSON (missing quotes, etc.)."""
        raw_output = '''{"chief_complaint": Headache, "duration": "2 days"}'''
        result = _parse_model_output(raw_output)
        
        assert "_error" in result
        assert result["chief_complaint"] == ""
        assert result["duration"] == ""
    
    def test_parse_no_json(self):
        """Test handling when no JSON is found in output."""
        raw_output = "I couldn't parse that."
        result = _parse_model_output(raw_output)
        
        assert "_error" in result
        assert "No JSON found" in result["_error"]
        assert all(result.get(k) == "" for k in ["chief_complaint", "duration", "severity", "history"])
    
    def test_parse_missing_fields(self):
        """Test that missing fields are filled with empty strings."""
        raw_output = '{"chief_complaint": "Fever", "severity": "Mild"}'
        result = _parse_model_output(raw_output)
        
        assert result["chief_complaint"] == "Fever"
        assert result["severity"] == "Mild"
        assert result["duration"] == ""  # filled
        assert result["history"] == ""   # filled

# ============================================================================
# INTEGRATION TESTS (these need the model or mock)
# ============================================================================

class TestStructureNoteIntegration:
    """Integration tests using the actual structure_note() function."""
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_simple_english_transcript(self):
        """Test with simple English input (baseline)."""
        transcript = "I have had a headache for 2 days. It's really bad."
        result = structure_note(transcript)
        
        # Assertions
        assert "_error" not in result, f"Error in result: {result.get('_error')}"
        assert result["chief_complaint"] != ""
        assert "headache" in result["chief_complaint"].lower()
        assert "2 days" in result["duration"] or "2 day" in result["duration"].lower()
        assert result["severity"].lower() in ["severe", "high", "very bad"]
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_hausa_transcript(self):
        """Test with Hausa language input."""
        # This would be a real Hausa transcript from ASR
        # For now, this is a placeholder
        transcript = "[Hausa transcript would go here]"
        result = structure_note(transcript)
        
        # Should be translated to English
        assert "_error" not in result or result["_error"] is None
        # All fields should be in English
        if result["chief_complaint"]:
            # Simple heuristic: English text should have common words
            pass  # TODO: add language detection if needed
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_no_hallucination(self):
        """Test that model doesn't invent details."""
        transcript = "I have a cough."
        result = structure_note(transcript)
        
        # Should NOT invent fever, shortness of breath, etc.
        # This is hard to test automatically; manual review recommended
        assert result.get("chief_complaint") is not None

# ============================================================================
# MOCK TESTS (don't require GPU)
# ============================================================================

class TestStructureNoteWithMock:
    """Test structure_note with mocked model output."""
    
    def test_structure_note_with_mock_error(self, monkeypatch):
        """Test error handling in structure_note."""
        # Mock _inference_local to raise an exception
        def mock_inference_error(*args, **kwargs):
            raise RuntimeError("Model loading failed")
        
        monkeypatch.setattr("nlp.structure_note._inference_local", mock_inference_error)
        
        result = structure_note("test transcript", use_local=True)
        
        assert "_error" in result
        assert "Model loading failed" in result["_error"]
        assert result["chief_complaint"] == ""

# ============================================================================
# PROMPT TESTING (manual — you iterate on prompts here)
# ============================================================================

class TestPromptQuality:
    """
    Manual tests for prompt quality.
    These are NOT automated — you run these and review output manually.
    
    Use these to verify:
    - Hausa/Igbo/Yoruba translation quality
    - No hallucination
    - JSON always valid
    - Handles code-switching
    """
    
    @pytest.mark.skip(reason="Manual review required")
    def test_prompt_hausa_example(self):
        """
        Manual test: Load a real Hausa transcript and check output quality.
        
        Instructions:
        1. Uncomment this test
        2. Replace [HAUSA_TRANSCRIPT] with real Hausa text from ASR
        3. Run: pytest tests/test_nlp_structuring.py::TestPromptQuality::test_prompt_hausa_example -v -s
        4. Review output manually
        5. If quality is bad, adjust CLINICAL_PROMPT_TEMPLATE and retry
        """
        transcript = "[HAUSA_TRANSCRIPT_HERE]"
        result = structure_note(transcript)
        
        print(f"\n\nHAUSA OUTPUT:\n{json.dumps(result, indent=2)}")
        
        # Manually verify:
        # - chief_complaint is not empty
        # - severity is one of: mild, moderate, severe
        # - no obvious hallucinations
        # - JSON is valid
        assert "_error" not in result

# ============================================================================
# CHECKLIST FOR YOUR DEVELOPMENT
# ============================================================================

"""
DEVELOPMENT CHECKLIST:

As you build structure_note.py, verify:

1. Model Loading
   [ ] Loads N-ATLaS from Hugging Face
   [ ] Works with 4-bit quantization (for 8GB GPUs)
   [ ] Works with fp16 (for 16GB GPUs)
   [ ] Works with Hugging Face Inference Endpoint (no local GPU)
   [ ] Caches model so it doesn't reload on every call

2. Inference
   [ ] Generates text with max_new_tokens cap (300)
   [ ] Uses temperature for sampling (not greedy)
   [ ] Handles long transcripts (truncate if needed)
   [ ] Times out gracefully if model hangs

3. Output Parsing
   [ ] Extracts JSON from raw output (even if it has preamble)
   [ ] Handles malformed JSON gracefully
   [ ] Always returns dict with 4 required keys + optional _error
   [ ] Never crashes on unexpected input

4. Translation Quality
   [ ] Translates Hausa to English
   [ ] Translates Igbo to English
   [ ] Translates Yoruba to English
   [ ] Handles code-switching (English + local language mixed)

5. Clinical Accuracy
   [ ] chief_complaint captures main symptom
   [ ] duration extracts timeframes correctly
   [ ] severity matches patient's description (not too high, not too low)
   [ ] history includes relevant background, not invented details
   [ ] No hallucinations (doesn't add symptoms patient didn't mention)

6. JSON Validity
   [ ] Output is ALWAYS valid JSON (no syntax errors)
   [ ] All 4 required fields present
   [ ] No extra keys unless _error
   [ ] Parses into Python dict cleanly

7. Integration
   [ ] Returns dict that Gradio can render
   [ ] Works with app.py's expected function signature
   [ ] Accepts any transcript length
   [ ] Handles empty/None inputs gracefully

8. Documentation
   [ ] Docstrings on all functions
   [ ] Example usage in __main__
   [ ] Comments on tricky sections
   [ ] STRUCTURING_GUIDE.md kept up to date

When all are checked, you're ready to hand off to the Frontend team!
"""
    def test_parse_json_with_preamble(self):
        """Test parsing JSON that has model preamble before it."""
        raw_output = '''You are a clinical assistant. Here's the structured note:
        {
            "chief_complaint": "Abdominal pain",
            "duration": "Since yesterday",
            "severity": "Severe",
            "history": "Patient reports vomiting"
        }
        Let me know if you need anything else.'''
        
        result = _parse_model_output(raw_output)
        assert result["chief_complaint"] == "Abdominal pain"
        assert "_error" not in result
    
    def test_parse_malformed_json(self):
        """Test handling of malformed JSON (missing quotes, etc.)."""
        raw_output = '''{"chief_complaint": Headache, "duration": "2 days"}'''
        result = _parse_model_output(raw_output)
        
        assert "_error" in result
        assert result["chief_complaint"] == ""
        assert result["duration"] == ""
    
    def test_parse_no_json(self):
        """Test handling when no JSON is found in output."""
        raw_output = "I couldn't parse that."
        result = _parse_model_output(raw_output)
        
        assert "_error" in result
        assert "No JSON found" in result["_error"]
        assert all(result.get(k) == "" for k in ["chief_complaint", "duration", "severity", "history"])
    
    def test_parse_missing_fields(self):
        """Test that missing fields are filled with empty strings."""
        raw_output = '{"chief_complaint": "Fever", "severity": "Mild"}'
        result = _parse_model_output(raw_output)
        
        assert result["chief_complaint"] == "Fever"
        assert result["severity"] == "Mild"
        assert result["duration"] == ""  # filled
        assert result["history"] == ""   # filled

# ============================================================================
# INTEGRATION TESTS (these need the model or mock)
# ============================================================================

class TestStructureNoteIntegration:
    """Integration tests using the actual structure_note() function."""
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_simple_english_transcript(self):
        """Test with simple English input (baseline)."""
        transcript = "I have had a headache for 2 days. It's really bad."
        result = structure_note(transcript)
        
        # Assertions
        assert "_error" not in result, f"Error in result: {result.get('_error')}"
        assert result["chief_complaint"] != ""
        assert "headache" in result["chief_complaint"].lower()
        assert "2 days" in result["duration"] or "2 day" in result["duration"].lower()
        assert result["severity"].lower() in ["severe", "high", "very bad"]
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_hausa_transcript(self):
        """Test with Hausa language input."""
        # This would be a real Hausa transcript from ASR
        # For now, this is a placeholder
        transcript = "[Hausa transcript would go here]"
        result = structure_note(transcript)
        
        # Should be translated to English
        assert "_error" not in result or result["_error"] is None
        # All fields should be in English
        if result["chief_complaint"]:
            # Simple heuristic: English text should have common words
            pass  # TODO: add language detection if needed
    
    @pytest.mark.skip(reason="Requires GPU or HF endpoint configured")
    def test_no_hallucination(self):
        """Test that model doesn't invent details."""
        transcript = "I have a cough."
        result = structure_note(transcript)
        
        # Should NOT invent fever, shortness of breath, etc.
        # This is hard to test automatically; manual review recommended
        assert result.get("chief_complaint") is not None

# ============================================================================
# MOCK TESTS (don't require GPU)
# ============================================================================

class TestStructureNoteWithMock:
    """Test structure_note with mocked model output."""
    
    def test_structure_note_with_mock_error(self, monkeypatch):
        """Test error handling in structure_note."""
        # Mock _inference_local to raise an exception
        def mock_inference_error(*args, **kwargs):
            raise RuntimeError("Model loading failed")
        
        monkeypatch.setattr("nlp.structure_note._inference_local", mock_inference_error)
        
        result = structure_note("test transcript", use_local=True)
        
        assert "_error" in result
        assert "Model loading failed" in result["_error"]
        assert result["chief_complaint"] == ""

# ============================================================================
# PROMPT TESTING (manual — you iterate on prompts here)
# ============================================================================

class TestPromptQuality:
    """
    Manual tests for prompt quality.
    These are NOT automated — you run these and review output manually.
    
    Use these to verify:
    - Hausa/Igbo/Yoruba translation quality
    - No hallucination
    - JSON always valid
    - Handles code-switching
    """
    
    @pytest.mark.skip(reason="Manual review required")
    def test_prompt_hausa_example(self):
        """
        Manual test: Load a real Hausa transcript and check output quality.
        
        Instructions:
        1. Uncomment this test
        2. Replace [HAUSA_TRANSCRIPT] with real Hausa text from ASR
        3. Run: pytest tests/test_nlp_structuring.py::TestPromptQuality::test_prompt_hausa_example -v -s
        4. Review output manually
        5. If quality is bad, adjust CLINICAL_PROMPT_TEMPLATE and retry
        """
        transcript = "[HAUSA_TRANSCRIPT_HERE]"
        result = structure_note(transcript)
        
        print(f"\n\nHAUSA OUTPUT:\n{json.dumps(result, indent=2)}")
        
        # Manually verify:
        # - chief_complaint is not empty
        # - severity is one of: mild, moderate, severe
        # - no obvious hallucinations
        # - JSON is valid
        assert "_error" not in result

# ============================================================================
# CHECKLIST FOR YOUR DEVELOPMENT
# ============================================================================

"""
DEVELOPMENT CHECKLIST:

As you build structure_note.py, verify:

1. Model Loading
   [ ] Loads N-ATLaS from Hugging Face
   [ ] Works with 4-bit quantization (for 8GB GPUs)
   [ ] Works with fp16 (for 16GB GPUs)
   [ ] Works with Hugging Face Inference Endpoint (no local GPU)
   [ ] Caches model so it doesn't reload on every call

2. Inference
   [ ] Generates text with max_new_tokens cap (300)
   [ ] Uses temperature for sampling (not greedy)
   [ ] Handles long transcripts (truncate if needed)
   [ ] Times out gracefully if model hangs

3. Output Parsing
   [ ] Extracts JSON from raw output (even if it has preamble)
   [ ] Handles malformed JSON gracefully
   [ ] Always returns dict with 4 required keys + optional _error
   [ ] Never crashes on unexpected input

4. Translation Quality
   [ ] Translates Hausa to English
   [ ] Translates Igbo to English
   [ ] Translates Yoruba to English
   [ ] Handles code-switching (English + local language mixed)

5. Clinical Accuracy
   [ ] chief_complaint captures main symptom
   [ ] duration extracts timeframes correctly
   [ ] severity matches patient's description (not too high, not too low)
   [ ] history includes relevant background, not invented details
   [ ] No hallucinations (doesn't add symptoms patient didn't mention)

6. JSON Validity
   [ ] Output is ALWAYS valid JSON (no syntax errors)
   [ ] All 4 required fields present
   [ ] No extra keys unless _error
   [ ] Parses into Python dict cleanly

7. Integration
   [ ] Returns dict that Gradio can render
   [ ] Works with app.py's expected function signature
   [ ] Accepts any transcript length
   [ ] Handles empty/None inputs gracefully

8. Documentation
   [ ] Docstrings on all functions
   [ ] Example usage in __main__
   [ ] Comments on tricky sections
   [ ] STRUCTURING_GUIDE.md kept up to date

When all are checked, you're ready to hand off to the Frontend team!
"""
