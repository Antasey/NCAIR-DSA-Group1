# LLM Structuring — Testing & Validation Checklist

Before you hand off the LLM module to the team, it needs to pass these tests.

---

## Setup Checklist

- [ ] `.env` file created with `NATLAS_ENDPOINT_URL` and `HF_TOKEN`
- [ ] Test that environment variables load: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('NATLAS_ENDPOINT_URL'))"`
- [ ] Install required packages: `pip install requests python-dotenv`
- [ ] Verify API endpoint is reachable (make a simple test call)

---

## Unit Tests

Run these in Python to validate the parsing and validation functions work.

### Test 1: JSON Extraction from Markdown

```python
from structure_note_IMPLEMENTATION import extract_json_from_text

# Test markdown code fence
input1 = """```json
{
  "chief_complaint": "Headache",
  "duration": "2 days",
  "severity": "moderate",
  "history": "None"
}
```"""

result = extract_json_from_text(input1)
assert '"chief_complaint"' in result, "Failed to extract JSON from markdown"
print("✓ Test 1 passed: Markdown extraction")
```

### Test 2: JSON with surrounding text

```python
input2 = "The structured note is: {\"chief_complaint\": \"Fever\", \"duration\": \"1 week\", \"severity\": \"severe\", \"history\": \"Flu-like symptoms\"}"

result = extract_json_from_text(input2)
assert '"fever"' in result.lower(), "Failed to extract embedded JSON"
print("✓ Test 2 passed: Embedded JSON extraction")
```

### Test 3: Field Validation

```python
from structure_note_IMPLEMENTATION import validate_structure

# Incomplete dict missing fields
incomplete = {"chief_complaint": "Pain"}

result = validate_structure(incomplete)
assert "duration" in result and result["duration"] == "Not mentioned"
assert "severity" in result and result["severity"] == "Not mentioned"
assert "history" in result and result["history"] == "Not mentioned"
print("✓ Test 3 passed: Field validation and filling")
```

### Test 4: Full Parsing Pipeline

```python
from structure_note_IMPLEMENTATION import parse_llm_output
import json

# Simulate LLM output with markdown
llm_output = """```json
{
  "chief_complaint": "Abdominal pain with vomiting",
  "duration": "Since yesterday",
  "severity": "severe",
  "history": "Nausea present"
}
```"""

result = parse_llm_output(llm_output)
assert result["chief_complaint"] == "Abdominal pain with vomiting"
assert result["severity"] == "severe"
assert "_error" not in result  # Should parse successfully
print("✓ Test 4 passed: Full parsing pipeline")
```

---

## Integration Tests

These test the full `structure_note()` function end-to-end.

### Test 5: Live LLM Call (English)

```python
from structure_note_IMPLEMENTATION import structure_note

transcript = "My head has been hurting for two days, it's a moderate pain, I took paracetamol but it didn't help"

result = structure_note(transcript)

assert result["chief_complaint"] != "", "No chief complaint extracted"
assert result["duration"] != "", "No duration extracted"
assert "two" in result["duration"].lower() or "2" in result["duration"], \
    f"Duration not correctly extracted: {result['duration']}"
assert result["severity"] in ["mild", "moderate", "severe"], \
    f"Severity not valid: {result['severity']}"

print("✓ Test 5 passed: English transcript structuring")
print(f"  Result: {result}")
```

### Test 6: Hausa Transcript (if you have a sample)

```python
# Example Hausa transcript (placeholder — use real if available)
hausa_transcript = "Kuna bugi jikina jiya, kuna abus, baba ne"  # "My body hurts since yesterday, it's bad"

result = structure_note(hausa_transcript)

# Check that output is in English
assert any(word in result["chief_complaint"].lower() 
           for word in ["pain", "hurt", "ache", "body"]), \
    f"Output doesn't appear to be translated: {result['chief_complaint']}"

print("✓ Test 6 passed: Hausa transcript translation + structuring")
print(f"  Result: {result}")
```

### Test 7: Empty/Invalid Input

```python
# Empty transcript
result = structure_note("")
assert "_error" in result or result["chief_complaint"] == ""
print("✓ Test 7a passed: Empty transcript handling")

# Very short transcript
result = structure_note("Pain")
assert result["chief_complaint"] != "" or "_error" in result
print("✓ Test 7b passed: Short transcript handling")
```

---

## Quality Checks

After running the above tests, manually verify the output quality:

### Check 1: Translation Accuracy

Run this test with a Hausa/Igbo/Yoruba sample (ask a native speaker to provide):

```python
result = structure_note("[hausa/igbo/yoruba transcript]")
print(json.dumps(result, indent=2))

# Manually verify:
# [ ] Output is clear English
# [ ] Medical terms preserved (e.g., "fever" not mistranslated)
# [ ] Patient's meaning captured accurately
```

### Check 2: No Hallucination

For this transcript (which deliberately omits details):
```python
transcript = "I have a headache"  # Very minimal

result = structure_note(transcript)

# Verify:
# [ ] history is "Not mentioned" (not invented)
# [ ] duration is "Not mentioned" (not inferred)
# [ ] severity is inferred BUT with low confidence
```

### Check 3: Consistency

Call the same transcript multiple times, check for consistency:

```python
transcript = "My stomach hurts since yesterday, very bad"

results = []
for i in range(3):
    result = structure_note(transcript)
    results.append(result)
    print(f"Run {i+1}: {result['chief_complaint']}")

# Verify all three runs produce same/similar output
```

---

## Performance Benchmark

Measure how long inference takes:

```python
import time
from structure_note_IMPLEMENTATION import structure_note

transcript = "I have had a fever for three days, it's quite severe, took paracetamol but didn't work"

start = time.time()
result = structure_note(transcript)
elapsed = time.time() - start

print(f"Inference time: {elapsed:.2f} seconds")

# Expected:
# - First call: 5-15 seconds (model warming up on endpoint)
# - Subsequent calls: 1-3 seconds
# If > 30 seconds: endpoint may be overloaded or timing out
```

---

## Error Handling Tests

### Test: Network Error

Temporarily disconnect internet or use a fake endpoint URL:

```python
os.environ["NATLAS_ENDPOINT_URL"] = "https://invalid-endpoint.com"

result = structure_note("My head hurts")

assert "_error" in result, "Error not captured"
assert result["chief_complaint"] == "", "Should return empty dict on error"
print("✓ Network error handling works")
```

### Test: Malformed LLM Response

Mock a bad LLM response:

```python
# Manually test parse_llm_output with bad JSON
from structure_note_IMPLEMENTATION import parse_llm_output

bad_json = "{ invalid json here }"

result = parse_llm_output(bad_json)

assert "_error" in result, "Should record error"
assert result["chief_complaint"] == "", "Should have empty fields"
print("✓ Malformed JSON handling works")
```

---

## Prompt Tuning Log

As you test with real examples, log what works and what doesn't. Use this template:

| Test Case | LLM Output | Issue | Fix Applied | Status |
|---|---|---|---|---|
| "Stomach pain since yesterday" | chief_complaint missing | LLM ignored field | Added "Extract EXACTLY these fields" | ✓ Pass |
| Hausa: "...ciwon..." | Not translated | LLM skipped translation | Added "Translate to English" | ✓ Pass |
| "Very bad pain" | severity: "very bad" | Not normalized | Changed prompt to specify "mild\|moderate\|severe" | ✓ Pass |

Keep this log and share findings with the team.

---

## Checklist for Code Review (Before Submitting)

- [ ] No hardcoded API keys (uses .env)
- [ ] Error handling on all API calls (try/except)
- [ ] Logging statements for debugging
- [ ] Type hints on functions
- [ ] Docstrings on all functions
- [ ] No unused imports
- [ ] Code formatted consistently (spaces, line length)
- [ ] Tests pass locally before pushing to GitHub

---

## Handoff Checklist

When you're ready to pass this to the Frontend team:

- [ ] Write a brief README in `nlp/README.md` explaining what `structure_note()` does
- [ ] Provide example usage:
  ```python
  from nlp.structure_note import structure_note
  
  transcript = "My head hurts"
  note = structure_note(transcript)
  print(note["chief_complaint"])  # Output: structured English note
  ```
- [ ] Verify `requirements.txt` includes: `requests`, `python-dotenv`
- [ ] Verify `.env.example` includes `NATLAS_ENDPOINT_URL` and `HF_TOKEN`
- [ ] Test that Frontend can import and call without errors
- [ ] Provide log of any known issues or limitations

---

## Common Issues & Fixes

### Issue: "NATLAS_ENDPOINT_URL not set"
**Fix:** Make sure `.env` file exists and has `NATLAS_ENDPOINT_URL=https://...`

### Issue: "Connection refused"
**Fix:** Endpoint URL is wrong or HF Inference Endpoint is down. Verify URL is correct.

### Issue: First call takes 60+ seconds
**Fix:** First request warms up the model on the endpoint. Subsequent calls are faster. Increase timeout from 30s to 60s.

### Issue: LLM returns non-JSON (e.g., "I cannot...")
**Fix:** Endpoint might be rejecting requests. Check HF_TOKEN is correct. May also mean model is overloaded.

### Issue: "chief_complaint" is empty in output
**Fix:** LLM didn't extract info. Try simpler prompt or check transcript clarity.

---

## When to Ask for Help

Post to the team Slack/chat if:

1. **First LLM call works, but parsing keeps failing** — Share raw LLM output, ask team to look at JSON format
2. **Endpoint is slow or timing out** — HF Inference Endpoint might need upgrade or team needs to add more replicas
3. **Language-specific issue** (e.g., Yoruba translation not working) — Test with native speakers, may need prompt adjustment
4. **Parsing is too strict** — If LLM output is slightly different, adjust regex in `extract_json_from_text()`

---

## Sign-Off

Once all tests pass:

```
Testing Status: ✓ READY FOR INTEGRATION

Signed: _________________ Date: _______

Tests Run: 7 unit + 3 integration + benchmarking
Success Rate: 100%
Known Issues: None
Performance: < 3s per call (average)
```
