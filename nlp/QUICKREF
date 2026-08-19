# LLM Structuring — Quick Reference Card

**Your Role:** Convert raw transcripts → structured JSON clinical notes using N-ATLaS

---

## Your Function Signature (DON'T CHANGE THIS)

```python
def structure_note(transcript: str) -> dict:
    """
    Input: raw patient speech (any language)
    Output: {"chief_complaint": "...", "duration": "...", "severity": "...", "history": "..."}
    """
```

---

## Quick Start (3 steps)

### 1. Install & Load Model
```bash
pip install transformers bitsandbytes torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "NCAIR1/N-ATLaS"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

### 2. Build Prompt
```python
prompt = """You are a clinical assistant. Convert this patient speech to JSON:
{transcript}

Respond with JSON only: {"chief_complaint": "...", "duration": "...", "severity": "...", "history": "..."}"""
```

### 3. Generate & Parse
```python
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=300)
response = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract JSON from response
import json, re
match = re.search(r'\{.*\}', response, re.DOTALL)
note = json.loads(match.group())
```

---

## Common Issues & Fixes

| Problem | Cause | Fix |
|---|---|---|
| Out of memory | Model too large for GPU | Use 4-bit quantization: `BitsAndBytesConfig(load_in_4bit=True)` |
| Invalid JSON | Model doesn't follow format | Add "Respond ONLY with JSON" to prompt |
| Hallucinated details | Model invents symptoms | Add "Do NOT invent details" + fewer examples |
| Hausa not translated | Model confused | Add "Translate to English" to prompt explicitly |
| Takes too long | Generating too many tokens | Reduce `max_new_tokens` from 300 to 200 |

---

## Prompt Tuning (THE MOST IMPORTANT THING)

Your prompt determines 90% of quality. Test these variations:

**Current (baseline):**
```
You are a clinical assistant. Extract patient speech into JSON.
[fields explained]
Do NOT invent details.
Respond ONLY with JSON.
```

**If model invents details:**
```
You are a clinical assistant. Extract ONLY information explicitly stated.
Use these exact fields: chief_complaint, duration, severity, history.
Do NOT add information the patient did not mention.
Respond ONLY with valid JSON.
```

**If code-switching fails:**
```
You are a clinical assistant. Patient may speak in Hausa, Igbo, or Yoruba.
Translate and extract into English JSON.
Only use what the patient explicitly said.
Respond ONLY with JSON in English.
```

---

## Testing

### Unit Test (no GPU needed)
```bash
pytest tests/test_nlp_structuring.py::TestJSONParsing -v
```

### Integration Test (GPU/endpoint needed)
```bash
pytest tests/test_nlp_structuring.py::TestStructureNoteIntegration -v
```

### Manual Test
```python
from nlp.structure_note import structure_note
result = structure_note("I have had a headache for 2 days")
print(result)
```

---

## Output Format (EXACT)

Always return dict with these keys:

```python
{
    "chief_complaint": "string",     # one-line symptom summary
    "duration": "string",             # "2 days", "since yesterday", etc.
    "severity": "string",             # "mild", "moderate", "severe"
    "history": "string",              # relevant background
    "_error": "string" (optional)      # only if something failed
}
```

---

## Success Metrics

- [ ] Consistent English output (all fields in English)
- [ ] Valid JSON every time (no parsing errors)
- [ ] No hallucinations (only patient's words)
- [ ] Handles Hausa/Igbo/Yoruba correctly
- [ ] Works with code-switching
- [ ] Output fits in review screen (not too long)

---

## Files You Own

```
nlp/
├── structure_note.py          ← MAIN FILE (you write this)
├── extract_keywords.py         (already exists — use as reference)
├── STRUCTURING_GUIDE.md        (detailed guide)
└── QUICKREF.md                 (this file)

tests/
└── test_nlp_structuring.py     (write tests here as you develop)
```

---

## When You're Done

1. ✅ All tests pass (unit + integration)
2. ✅ Tested with real Hausa/Igbo/Yoruba samples
3. ✅ No JSON parsing errors
4. ✅ No hallucinations
5. ✅ Ready to integrate with Gradio (Frontend team)

**Hand off:** Give the Frontend team your completed `structure_note.py` + `STRUCTURING_GUIDE.md` + test results.

---

## Resources

- N-ATLaS model: https://huggingface.co/NCAIR1/N-ATLaS
- Transformers docs: https://huggingface.co/docs/transformers/
- Prompt engineering: https://platform.openai.com/docs/guides/prompt-engineering
