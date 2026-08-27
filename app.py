"""
NCAIR-DSA — Gradio entry point.
Pipeline: audio capture -> ASR -> LLM structuring -> keyword extraction -> 
review (full note editable, keywords + causes/effects as reference sidebar) ->
patient record save.

Main goal: full structured English clinical note (removes translation barrier).
Keywords + causes/effects: supporting quick-reference sidebar for doctor during review.
"""
import gradio as gr
import json
from nlp.extract_keywords import extract_keywords, get_causes_and_effects
# from asr.transcribe import transcribe_audio
# from nlp.structure_note import structure_note
# from templates.clinical_note import save_patient_record

def process_patient_intake(patient_id, language, audio_file):
    """
    End-to-end pipeline:
    1. Transcribe audio to text
    2. Structure into full English clinical note (main deliverable)
    3. Extract keywords from transcript (supporting reference)
    4. Look up possible causes/effects per detected symptom (reference only)
    5. Return: full note (editable) + keywords + causes/effects for doctor review
    """
    if not audio_file or not patient_id:
        return "", "", "", "", "Error: Patient ID and audio required"

    # TODO: wire up transcription
    # transcript = transcribe_audio(audio_file, language)
    transcript = "[Transcription placeholder]"

    # TODO: wire up structuring — this is the MAIN note
    # note = structure_note(transcript)
    note = {
        "chief_complaint": "Patient reports abdominal pain",
        "duration": "Since yesterday",
        "severity": "Moderate to severe",
        "history": "No previous similar episodes mentioned"
    }

    # Extract keywords from raw transcript (supporting reference only)
    keywords = extract_keywords(transcript)

    # Look up reference causes/effects for detected symptoms (decision-support only)
    causes_effects = get_causes_and_effects(keywords.get("symptoms", []))

    # Format full note as readable text (doctor edits this)
    full_note_text = f"""CLINICAL NOTE
Patient ID: {patient_id}
Language: {language}
Date: [Auto-filled]

Chief Complaint:
{note.get('chief_complaint', '')}

Duration:
{note.get('duration', '')}

Severity:
{note.get('severity', '')}

Relevant History:
{note.get('history', '')}

Raw Transcript (for reference):
{transcript}
"""

    # Format keywords as reference sidebar
    keywords_text = f"""QUICK REFERENCE KEYWORDS
(Do not edit — for doctor scanning during review)

Symptoms Detected:
{', '.join(keywords.get('symptoms', [])) if keywords.get('symptoms') else 'None'}

Duration:
{', '.join(keywords.get('duration', [])) if keywords.get('duration') else 'None'}

Severity Level:
{', '.join(keywords.get('severity', [])) if keywords.get('severity') else 'None'}

Anatomical Sites:
{', '.join(keywords.get('anatomical_sites', [])) if keywords.get('anatomical_sites') else 'None'}
"""

    # Format causes/effects as a reference panel (per symptom)
    if causes_effects:
        blocks = []
        for symptom, info in causes_effects.items():
            causes = ', '.join(info.get('possible_causes', [])) or 'None listed'
            effects = ', '.join(info.get('possible_effects', [])) or 'None listed'
            blocks.append(
                f"● {symptom.title()}\n"
                f"  Possible Causes: {causes}\n"
                f"  Possible Effects: {effects}"
            )
        causes_effects_text = (
            "POSSIBLE CAUSES & EFFECTS (Reference Only — Verify Clinically)\n\n"
            + "\n\n".join(blocks)
        )
    else:
        causes_effects_text = (
            "POSSIBLE CAUSES & EFFECTS (Reference Only — Verify Clinically)\n\n"
            "No matched symptoms to look up yet."
        )

    # Prepare patient record with editable note + reference keywords + causes/effects
    patient_record = {
        "patient_id": patient_id,
        "language": language,
        "raw_transcript": transcript,
        "extracted_keywords": keywords,
        "causes_and_effects": causes_effects,
        "full_clinical_note": note,
        "status": "ready_for_review"
    }

    return full_note_text, keywords_text, causes_effects_text, json.dumps(patient_record, indent=2), "✓ Ready for review"

def save_patient_record(patient_id, edited_note_text):
    """Save the reviewed/edited note to the database."""
    try:
        # TODO: wire up database save with edited_note_text
        # save_patient_record(patient_id, edited_note_text)
        return f"✓ Patient {patient_id} record saved successfully"
    except Exception as e:
        return f"✗ Error saving record: {str(e)}"

# Gradio Interface
with gr.Blocks(title="NCAIR-DSA: Patient Intake Assistant") as demo:
    gr.Markdown("# Patient Intake Assistant\n**Multi-lingual voice-to-clinical-note system**\n\nRemoves translation barriers. Keywords and causes/effects provide quick reference during doctor review.")

    # Input Section
    gr.Markdown("## Step 1: Record Patient Audio")
    with gr.Row():
        patient_id = gr.Textbox(label="Patient ID", placeholder="e.g., P001", scale=1)
        language = gr.Dropdown(choices=["Hausa", "Igbo", "Yoruba"], label="Language", scale=1)

    audio_input = gr.Audio(label="Patient Audio (Upload or Record)", type="filepath", sources=["upload", "microphone"])

    transcribe_btn = gr.Button("Transcribe & Generate Clinical Note", variant="primary", size="lg")

    # Review Section
    gr.Markdown("## Step 2: Doctor Review & Edit")
    gr.Markdown("**Left: Full clinical note (editable).** | **Right: Keywords + possible causes/effects (read-only sidebar).**")

    with gr.Row():
        # Main column: full editable note
        with gr.Column(scale=3):
            gr.Markdown("### Full Clinical Note (Edit as Needed)")
            full_note = gr.Textbox(
                label="Doctor edits this before saving — removes translation barrier",
                lines=20,
                interactive=True,
                placeholder="Structured English clinical note will appear here..."
            )

        # Sidebar: keywords + causes/effects reference (non-editable)
        with gr.Column(scale=1):
            gr.Markdown("### Quick Keywords (Reference Only)")
            keywords_display = gr.Textbox(
                label="For scanning during review — read only",
                lines=12,
                interactive=False,
                placeholder="Keywords will appear here..."
            )
            gr.Markdown("### Possible Causes & Effects (Reference Only)")
            causes_effects_display = gr.Textbox(
                label="Not a diagnosis — verify clinically before acting",
                lines=14,
                interactive=False,
                placeholder="Possible causes and effects per symptom will appear here..."
            )

    # Save Section
    gr.Markdown("## Step 3: Save to Patient History")
    save_btn = gr.Button("Save Patient Record to Database", variant="primary", size="lg")
    save_status = gr.Textbox(label="Status", interactive=False, show_label=True)

    # Hidden: full patient record JSON (for export if needed)
    with gr.Accordion("Debug: Full Patient Record (JSON)", open=False):
        patient_record_json = gr.Textbox(label="Patient Record JSON", lines=10, interactive=False)

    # Wire up the pipeline
    transcribe_btn.click(
        fn=process_patient_intake,
        inputs=[patient_id, language, audio_input],
        outputs=[full_note, keywords_display, causes_effects_display, patient_record_json, save_status]
    )

    save_btn.click(
        fn=save_patient_record,
        inputs=[patient_id, full_note],
        outputs=[save_status]
    )

if __name__ == "__main__":
    demo.launch()
