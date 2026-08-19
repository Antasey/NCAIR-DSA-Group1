# LLM Structuring Module — Complete Guide

**Your job:** Take raw patient transcripts (in any of the three languages) and turn them into structured English clinical notes using N-ATLaS LLM.

---

## What You're Building

**Input:** Raw patient speech transcript (Yoruba/Igbo/Hausa)
```
"Stomach pain started yesterday, I've been vomiting too, it's really bad"
```

**Output:** Structured JSON (English)
```json
{
  "chief_complaint": "Abdominal pain with vomiting",
  "duration": "Since yesterday",
  "severity": "Severe",
  "history": "Patient reports nausea alongside pain"
}
```

This structured note is what the doctor sees and edits — it's the **main deliverable** that removes the language barrier.

---

## N-ATLaS LLM Model

**Model ID:** `NCAIR1/N-ATLaS`
- **Size:** 8B parameters (Llama-3 base)
- **Task:** Multilingual instruction-following (translate + structure + summarize)
- **Hardware needed:**
  - Full precision (fp32): ~32GB VRAM (not realistic)
  - Half precision (fp16): ~16GB VRAM (one good GPU)
  - 4-bit quantization: ~6-8GB VRAM (mid-range GPU or Colab)
  - **Hosted option:** Use Hugging Face Inference Endpoint (recommended for team) — no local GPU needed

---

## Implementation Strategy

### Option A: Hosted Inference Endpoint (RECOMMENDED FOR TEAM)

**Pros:**
- No GPU needed locally
- Faster for prototyping
- Easy to share with team
- Scalable

**Setup:**
1. Create HF Inference Endpoint: `https://huggingface.co/inference-endpoints`
2. Select `NCAIR1/N-ATLaS` model
3. Set to private (only accessible with API key)
4. Get endpoint URL + API key
5. Store in `.env`: `NATLAS_ENDPOINT_URL` and `HF_TOKEN`

**Code:**
```python
import os
import requests
import json

ENDPOINT_URL = os.getenv("NATLAS_ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

def call_natlas_endpoint(prompt: str) -> str:
    """Call hosted N-ATLaS LLM via Hugging Face Inference Endpoint."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3,  # Low temp for consistent structure
            "top_p": 0.9
        }
    }
    response = requests.post(ENDPOINT_URL, headers=headers, json=payload)
    result = response.json()
    return result[0]["generated_text"]
```

### Option B: Local GPU (If available)

**Pros:**
- Full control
- No API calls
- Faster inference once loaded

**Requirements:**
- GPU with 8GB+ VRAM (RTX 3060 or better)
- `bitsandbytes` for 4-bit quantization

**Code:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "NCAIR1/N-ATLaS"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    load_in_4bit=True,  # 4-bit quantization to save VRAM
    bnb_4bit_compute_dtype=torch.float16
)

def call_natlas_local(prompt: str) -> str:
    """Call N-ATLaS LLM locally."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## Prompt Engineering

This is the critical part. The prompt tells the LLM exactly what to do.

### Key Principles
1. **Be explicit:** Tell it to output JSON, not free text
2. **Give examples:** Show the format you want
3. **Set boundaries:** "Do not invent information"
4. **Handle all three languages:** Mention you want English output regardless of input language

### Template

```python
STRUCTURE_PROMPT = """You are a clinical note assistant. Your job is to take a patient's raw speech 
(in Yoruba, Igbo, Hausa, or English) and structure it into a clean English clinical note.

INSTRUCTIONS:
1. Translate the patient's speech into English if needed
2. Extract and organize these four fields:
   - chief_complaint: What the patient's main problem is (1-2 sentences)
   - duration: How long they've had the problem (e.g., "since yesterday", "3 days")
   - severity: How bad the problem is (mild, moderate, severe)
   - history: Any other relevant medical details they mentioned

3. Use ONLY information the patient actually said. Do NOT invent details.
4. Respond with valid JSON only, no other text.

EXAMPLE:
Patient speech: "My head hurts a lot since two days ago, I also feel sick"
Output:
{{
  "chief_complaint": "Headache with nausea",
  "duration": "2 days",
  "severity": "severe",
  "history": "Patient reports nausea accompanying the headache"
}}

Now structure this patient's speech:
Patient speech: {transcript}
Output (JSON only):"""
```

### Tuning the Prompt

Test these variations and see which produces the most consistent output:

**Variation A: Strictest (for messy transcripts)**
```
Respond ONLY with valid JSON. Do not include any explanatory text before or after the JSON.
```

**Variation B: With field descriptions**
```
- chief_complaint: The patient's primary symptom or complaint (required, max 10 words)
- duration: Time period since onset (required, format like "2 days", "since yesterday")
- severity: Scale from mild to severe based on patient's language (required)
- history: Any secondary symptoms or relevant past medical info mentioned (can be empty)
```

**Variation C: Anti-hallucination (strongest)**
```
CRITICAL: Only include information the patient explicitly stated. If the patient did not 
mention a field, set it to "Not mentioned" rather than guessing or inferring.
```

---

## JSON Parsing & Validation

The LLM might:
- Return JSON wrapped in markdown backticks: ` ```json {...} ``` `
- Return partial JSON (incomplete)
- Return non-JSON text

You need defensive parsing:

```python
import json
import re

def parse_llm_output(raw_output: str) -> dict:
    """Extract and validate JSON from LLM output."""
    
    # Remove markdown code fences if present
    cleaned = raw_output.replace("```json", "").replace("```", "").strip()
    
    # Try to extract JSON if embedded in text
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group()
    
    try:
        data = json.loads(cleaned)
        # Validate required fields
        required = ["chief_complaint", "duration", "severity", "history"]
        for field in required:
            if field not in data:
                data[field] = ""
        return data
    except json.JSONDecodeError as e:
        # Fallback: return empty structure
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": f"JSON parse failed: {str(e)}",
            "_raw": raw_output
        }
```

---

## Full Implementation

```python
"""
LLM Structuring Stage — converts raw transcripts into structured clinical notes
using the N-ATLaS LLM model for translation, formatting, and extraction.
"""
import os
import json
import re
import requests
from typing import Dict

# Choose your mode
MODE = os.getenv("LLM_MODE", "endpoint")  # "endpoint" or "local"

if MODE == "endpoint":
    ENDPOINT_URL = os.getenv("NATLAS_ENDPOINT_URL")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not ENDPOINT_URL:
        raise ValueError("NATLAS_ENDPOINT_URL not set in .env")

STRUCTURE_PROMPT = """You are a clinical note assistant. Your job is to take a patient's raw speech 
(in Yoruba, Igbo, Hausa, or English) and structure it into a clean English clinical note.

INSTRUCTIONS:
1. Translate the patient's speech into English if it's in another language
2. Extract and organize these four fields:
   - chief_complaint: What the patient's main problem is (1-2 sentences)
   - duration: How long they've had the problem (e.g., "since yesterday", "3 days")
   - severity: How bad the problem is (mild, moderate, or severe)
   - history: Any other relevant medical details they mentioned

3. Use ONLY information the patient actually said. Do NOT invent details.
4. If a field is not mentioned, set it to "Not mentioned"
5. Respond with ONLY valid JSON, no other text before or after.

EXAMPLE:
Patient: "My stomach hurts a lot since two days ago, I've been vomiting"
{{
  "chief_complaint": "Abdominal pain with vomiting",
  "duration": "Since 2 days ago",
  "severity": "severe",
  "history": "Patient reports nausea with the pain"
}}

Now structure this patient's speech:
Patient: {transcript}

Respond with JSON only:"""

def call_llm_endpoint(prompt: str) -> str:
    """Call N-ATLaS via Hugging Face Inference Endpoint."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.3,  # Low temp = more consistent structure
            "top_p": 0.9
        }
    }
    try:
        response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=30)
        result = response.json()
        return result[0]["generated_text"]
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def parse_json_output(raw_output: str) -> dict:
    """Extract and validate JSON from LLM output."""
    cleaned = raw_output.replace("```json", "").replace("```", "").strip()
    
    # Try to extract JSON if embedded in text
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group()
    
    try:
        data = json.loads(cleaned)
        # Ensure all required fields exist
        required = ["chief_complaint", "duration", "severity", "history"]
        for field in required:
            if field not in data:
                data[field] = "Not mentioned"
        return data
    except json.JSONDecodeError:
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "JSON parsing failed",
            "_raw": raw_output[:200]  # Store first 200 chars for debugging
        }

def structure_note(transcript: str) -> dict:
    """
    Main function: convert raw patient transcript into structured clinical note.
    
    Input: Raw patient speech (any language)
    Output: Dict with chief_complaint, duration, severity, history (English)
    """
    if not transcript or len(transcript.strip()) == 0:
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "Empty transcript"
        }
    
    # Build prompt
    prompt = STRUCTURE_PROMPT.format(transcript=transcript)
    
    # Call LLM
    if MODE == "endpoint":
        raw_output = call_llm_endpoint(prompt)
    else:
        raise NotImplementedError("Local mode not yet implemented")
    
    # Parse and return
    structured = parse_json_output(raw_output)
    return structured

def structure_note_batch(transcripts: list) -> list:
    """Process multiple transcripts at once (useful for testing)."""
    return [structure_note(t) for t in transcripts]
```

---

## Testing

Before integrating into the app, test this locally:

```python
test_cases = [
    "My stomach has been paining me since yesterday, I've been vomiting too, very bad",
    "Headache for 3 days, mild, took paracetamol but didn't help",
    "Cough started last week, getting worse, no fever"
]

for transcript in test_cases:
    result = structure_note(transcript)
    print(f"Input: {transcript}")
    print(f"Output: {json.dumps(result, indent=2)}\n")
```

---

## Deployment Checklist

Before shipping to the team:

- [ ] `.env.example` includes `NATLAS_ENDPOINT_URL` and `HF_TOKEN`
- [ ] Error handling for network/timeout issues
- [ ] Logging of LLM calls (for debugging)
- [ ] Unit tests for JSON parsing edge cases
- [ ] Documentation of prompt tuning experiments
- [ ] Performance benchmark (how long does inference take?)

---

## Common Issues & Fixes

**Issue:** LLM returns markdown code fence (` ```json {...} ``` `)
**Fix:** Already handled in `parse_json_output()` — removes backticks

**Issue:** LLM hallucinates fields not in transcript
**Fix:** Add to prompt: "Use ONLY information the patient actually said"

**Issue:** Timeout on first call (model warming up)
**Fix:** Increase timeout: `requests.post(..., timeout=60)`

**Issue:** Inconsistent severity levels (sometimes "very severe", sometimes "moderate")
**Fix:** Add to prompt: `severity: one of (mild, moderate, severe)`

---

## Next Steps

1. **Set up HF Inference Endpoint** with `NCAIR1/N-ATLaS`
2. **Store endpoint URL & token in `.env`**
3. **Test `structure_note()` with real Hausa/Yoruba/Igbo transcripts**
4. **Wire it into `app.py`** in the `process_patient_intake()` function
5. **Iterate prompt** based on what the team gives you (test with real clinic data)

The **LLM structuring is the core of this project** — spend time tuning the prompt with real examples from your pilot site. A good prompt beats a better model.
