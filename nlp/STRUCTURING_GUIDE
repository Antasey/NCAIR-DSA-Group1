# LLM Structuring Module — N-ATLaS Guide

You're responsible for converting raw patient speech transcripts (from ASR) into 
structured English clinical notes using the N-ATLaS LLM. This is the CORE of the 
project — everything else depends on your output quality.

## Your Job

**Input:** Raw transcript in Yoruba/Igbo/Hausa (or mixed with English)
Example: "My stomach has been paining me since yesterday evening and I've been vomiting"

**Output:** Structured English clinical note (JSON format)
```json
{
  "chief_complaint": "Abdominal pain with vomiting",
  "duration": "Since yesterday evening",
  "severity": "Moderate to severe",
  "history": "Patient reports onset yesterday with accompanying nausea"
}
```

## The Model: N-ATLaS-LLM

- **Model:** `NCAIR1/N-ATLaS` on Hugging Face
- **Type:** Fine-tuned Llama-3 8B (autoregressive text generation)
- **Training:** Instruction-tuned on ~400M tokens of multilingual Nigerian language data
- **Size:** 8 billion parameters
- **Context:** 8,092 tokens

### What it's good at:
- Translation from Yoruba/Igbo/Hausa to English
- Following structured output instructions (JSON)
- Medical/clinical terminology

### What it struggles with:
- Code-switching (patient mixing English + local language mid-sentence)
- Very long transcripts (over 8K tokens)
- Maintaining consistency if prompt isn't tight enough
- Extracting information not explicitly stated (hallucination risk)

## Hardware Requirements

**Option A: Local GPU (Recommended for development)**
- RTX 4090 / A100: ~32GB VRAM (fp32) — not necessary
- RTX 3090 / A10: ~16GB VRAM (fp16) — comfortable
- RTX 4060 / A4000: ~8GB VRAM (4-bit quantization) — tight but works
- Command: `python structure_note.py --local`

**Option B: Google Colab (Free, limited)**
- Standard GPU gives you T4 (16GB) — fp16 works
- Command: `python structure_note.py --colab`

**Option C: Hugging Face Inference Endpoint (Hosted, costs $)**
- No local GPU needed
- API call pattern (like calling a web service)
- Command: `python structure_note.py --endpoint <URL>`

## Prompting Strategy

The prompt is THE MOST IMPORTANT THING. A good prompt gets 90% accuracy; a bad one 
gets 30%. Your job is to design and test prompts.

### Core Prompt Template

```
You are a clinical assistant. Convert the following patient speech into a structured 
clinical note in English. Extract and organize the information into these fields:
- chief_complaint: one-line summary of why they came
- duration: how long they've had the symptom (e.g., "3 days", "since yesterday")
- severity: mild, moderate, or severe
- history: relevant background the patient mentions

IMPORTANT RULES:
1. Do NOT invent details. Only use what the patient actually said.
2. If a field has no information, leave it empty but include the key.
3. Respond ONLY with valid JSON, no other text.

Patient transcript: {transcript}

Respond with JSON only:
```

### Why this works:
- **Clear task:** "Convert patient speech to structured note"
- **Specific fields:** doctor knows what to expect
- **Rules:** prevents hallucination, forces JSON
- **No preamble:** "Respond with JSON only" stops the model talking before the JSON

## Loading & Running the Model

### Local GPU (4-bit quantized)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load with 4-bit quantization to fit on smaller GPUs
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model_id = "NCAIR1/N-ATLaS"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Inference
prompt = f"You are a clinical assistant...\n\nPatient transcript: {transcript}"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=300, temperature=0.7, top_p=0.95)
response = tokenizer.decode(output[0], skip_special_tokens=True)
```

### Hugging Face Inference Endpoint (No local GPU)

```python
import requests
import json

endpoint_url = "https://your-hf-endpoint.com"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

payload = {
    "inputs": f"You are a clinical assistant...\n\nPatient transcript: {transcript}",
    "parameters": {
        "max_new_tokens": 300,
        "temperature": 0.7,
        "top_p": 0.95
    }
}

response = requests.post(endpoint_url, headers=headers, json=payload)
result = response.json()
generated_text = result[0]["generated_text"]
```

## Output Parsing

The model will generate text like:

```
You are a clinical assistant...
{
  "chief_complaint": "Abdominal pain",
  "duration": "Since yesterday",
  "severity": "Severe",
  "history": "No previous episodes"
}
```

You need to extract just the JSON:

```python
import json
import re

def parse_model_output(raw_output):
    # Find JSON block in the output
    match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # If JSON is malformed, return error dict
            return {
                "chief_complaint": "",
                "duration": "",
                "severity": "",
                "history": "",
                "_error": "JSON parsing failed",
                "_raw": raw_output
            }
    else:
        return {
            "chief_complaint": "",
            "duration": "",
            "severity": "",
            "history": "",
            "_error": "No JSON found in output",
            "_raw": raw_output
        }
```

## Testing Your Code

### Test case 1: Simple English (baseline)
Input: "I have had a headache for 2 days. It's really bad."
Expected output:
```json
{
  "chief_complaint": "Headache",
  "duration": "2 days",
  "severity": "Severe",
  "history": ""
}
```

### Test case 2: Hausa speech (from ASR simulation)
Input: "[Hausa transcript from ASR model]"
Expected: Structured note in English

### Test case 3: Code-switching (hard case)
Input: "My back has been paining me since last week and it's very bad"
Expected: Clean English output despite code-switching in input

## Integration with the Pipeline

Your code gets called from `app.py`:

```python
from nlp.structure_note import structure_note

transcript = "[ASR output from asr/transcribe.py]"
note = structure_note(transcript)  # returns dict
# Then the dict gets rendered in the review screen
```

So your function signature MUST be:

```python
def structure_note(transcript: str) -> dict:
    """
    Convert a raw patient transcript to structured clinical note.
    
    Args:
        transcript (str): Raw transcript from ASR (may be in Yoruba/Igbo/Hausa)
    
    Returns:
        dict: Structured note with keys:
            - chief_complaint (str)
            - duration (str)
            - severity (str)
            - history (str)
            - _error (str, optional): error message if parsing failed
    """
    pass
```

## Prompt Tuning Checklist

As you iterate on your prompt, test against these axes:

- [ ] Does it translate Hausa correctly? (test with a Hausa sample)
- [ ] Does it translate Igbo correctly?
- [ ] Does it translate Yoruba correctly?
- [ ] Does it handle code-switching (English + local language)?
- [ ] Does it stay concise (not hallucinating extra details)?
- [ ] Does it handle missing information (leaves fields empty, doesn't guess)?
- [ ] Does it output valid JSON 100% of the time?
- [ ] Is the JSON parsing robust enough for malformed output?
- [ ] Does it work with both short and long transcripts?

## Common Pitfalls

1. **Hallucinating details:** Prompt says "Do NOT invent" but model still does. Solution: Use "ONLY use information explicitly stated by the patient" and penalize at inference time.

2. **Breaking JSON:** Model outputs `"severity": Severe,` (missing quotes). Solution: Post-process and fix quotes, or use stricter prompting.

3. **Too long output:** Model generates 2K tokens when you asked for 300. Solution: Tighten `max_new_tokens`, use `stop_sequences` if the API supports it.

4. **Ignoring language:** Model translates Hausa but keeps Hausa words in output. Solution: Add "Respond ONLY in English" to the prompt.

5. **Forgetting to strip model preamble:** You get `"You are a clinical assistant. [JSON]"` — extract just the JSON.

## Success Criteria

Your module is done when:

- ✅ Consistently converts Hausa/Igbo/Yoruba transcripts to English clinical notes
- ✅ Output is always valid JSON
- ✅ No hallucinated details
- ✅ Handles code-switching gracefully
- ✅ Integrates cleanly with the ASR input and Gradio output
- ✅ Works with both local GPU and HF Inference Endpoint
- ✅ Documented code with examples

## Resources

- N-ATLaS model card: https://huggingface.co/NCAIR1/N-ATLaS
- Transformers library: https://huggingface.co/docs/transformers/
- Prompt engineering guide: https://platform.openai.com/docs/guides/prompt-engineering
- 4-bit quantization (bitsandbytes): https://huggingface.co/docs/bitsandbytes/

## Next Steps

1. Read the N-ATLaS model card carefully
2. Load the model locally or on Colab
3. Test with sample Hausa/Igbo/Yoruba text
4. Iterate on the prompt until output is clean
5. Write tests in `tests/test_nlp_structuring.py`
6. Document your prompt tuning decisions
7. Hand off to the Frontend team with clear documentation
