"""
NCAIR-DSA — Gradio entry point.
Pipeline: audio capture -> ASR -> LLM structuring -> keyword extraction ->
review (individual fields editable, keywords as reference sidebar) -> patient record save.

Main goal: full structured English clinical note (removes translation barrier).
Keywords: supporting quick-reference sidebar for doctor during review.

Fields are kept SEPARATE (not one text blob) so what the doctor edits maps
directly to what gets saved in the database — no re-parsing free text.
"""
import gradio as gr
import json
# Mount Google Drive BEFORE importing clinical_note — that module checks
# whether Drive is mounted to decide where the database lives.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass  # not running in Colab — falls back to local storage

from asr.transcribe import transcribe_audio
from nlp.structure_note import structure_note
from nlp.extract_keywords import extract_keywords
from templates.clinical_note import init_db, save_patient_record as db_save_patient_record, get_patient_history

# Ensure database tables exist on startup
init_db()


def process_patient_intake(patient_id, language, audio_file):
    """
    End-to-end pipeline:
    1. Transcribe audio to text
    2. Structure into full English clinical note (main deliverable) — separate fields
    3. Extract keywords from transcript (supporting reference)
    4. Return: individual editable fields + keywords sidebar + raw transcript (hidden state)

    Returns (in order matching Gradio outputs):
        chief_complaint, duration, severity, history,
        keywords_display_text, transcript_state, keywords_state, status_message
    """
    if not audio_file or not patient_id:
        return "", "", "", "", "", "", "{}", "Error: Patient ID and audio required"

    # --- Step 1: Transcribe ---
    try:
        transcript = transcribe_audio(audio_file, language)
    except Exception as e:
        return "", "", "", "", "", "", "{}", f"Error transcribing audio: {str(e)}"

    # --- Step 2: Structure into clinical note (MAIN deliverable) ---
    note = structure_note(transcript)

    if note.get("_error"):
        status = f"⚠ Note generated with issues: {note['_error']}"
    else:
        status = "✓ Ready for review"

    # --- Step 3: Extract keywords from the ENGLISH structured note ---
    # (not the raw transcript, which may be in Hausa/Igbo/Yoruba — the
    # keyword list is English words, so it needs English text to match against.
    # The doctor is reading English, so keywords must come from what they read.)
    english_note_text = " ".join([
        note.get("chief_complaint", ""),
        note.get("duration", ""),
        note.get("severity", ""),
        note.get("history", ""),
    ])
    keywords = extract_keywords(english_note_text)

    keywords_text = f"""Symptoms Detected:
{', '.join(keywords.get('symptoms', [])) if keywords.get('symptoms') else 'None'}

Duration Mentions:
{', '.join(keywords.get('duration', [])) if keywords.get('duration') else 'None'}

Severity Language:
{', '.join(keywords.get('severity', [])) if keywords.get('severity') else 'None'}

Anatomical Sites:
{', '.join(keywords.get('anatomical_sites', [])) if keywords.get('anatomical_sites') else 'None'}
"""

    return (
        note.get("chief_complaint", ""),
        note.get("duration", ""),
        note.get("severity", ""),
        note.get("history", ""),
        keywords_text,
        transcript,                    # hidden state — needed at save time
        json.dumps(keywords),          # hidden state — needed at save time
        status,
    )


def handle_save_click(patient_id, language, chief_complaint, duration, severity, history,
                       transcript_state, keywords_state_json):
    """
    Gradio click handler for saving the (doctor-reviewed) record to the database.
    Renamed to avoid shadowing templates.clinical_note.save_patient_record.
    """
    if not patient_id:
        return "Error: Patient ID is required to save"

    try:
        keywords = json.loads(keywords_state_json) if keywords_state_json else {}

        db_save_patient_record(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            duration=duration,
            severity=severity,
            history=history,
            language=language,
            keywords=keywords,
            transcript=transcript_state,
        )
        return f"✓ Patient {patient_id} record saved successfully"
    except Exception as e:
        return f"✗ Error saving record: {str(e)}"


def handle_view_history_click(patient_id):
    """Look up and display a patient's past visits."""
    if not patient_id:
        return "Enter a Patient ID to view history"

    visits = get_patient_history(patient_id)

    if not visits:
        return f"No previous visits found for patient {patient_id}"

    lines = [f"History for {patient_id} ({len(visits)} visit(s)):\n"]
    for visit in visits:
        visit_id, chief_complaint, duration, severity, history, language, created_at = visit
        lines.append(
            f"— {created_at} [{language}]\n"
            f"  Complaint: {chief_complaint}\n"
            f"  Duration: {duration} | Severity: {severity}\n"
            f"  History: {history}\n"
        )
    return "\n".join(lines)


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="NCAIR-DSA: Patient Intake Assistant") as demo:
    gr.Markdown(
        "# Patient Intake Assistant\n"
        "**Multi-lingual voice-to-clinical-note system**\n\n"
        "Removes translation barriers. Keywords provide quick reference during doctor review."
    )

    # --- Hidden state: carries raw transcript + keywords through to save ---
    transcript_state = gr.State("")
    keywords_state = gr.State("{}")

    # Step 1: Capture
    gr.Markdown("## Step 1: Record Patient Audio")
    with gr.Row():
        patient_id = gr.Textbox(label="Patient ID", placeholder="e.g., P001", scale=1)
        language = gr.Dropdown(choices=["Hausa", "Igbo", "Yoruba"], label="Language", scale=1)

    audio_input = gr.Audio(
        label="Patient Audio (Upload or Record)",
        type="filepath",
        sources=["upload", "microphone"]
    )

    transcribe_btn = gr.Button("Transcribe & Generate Clinical Note", variant="primary", size="lg")
    process_status = gr.Textbox(label="Status", interactive=False)

    # Step 2: Review — separate editable fields (not one text blob)
    gr.Markdown("## Step 2: Doctor Review & Edit")
    gr.Markdown(
        "**Left: Full clinical note fields (editable).** | "
        "**Right: Quick reference keywords (read-only sidebar).**"
    )

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Full Clinical Note (Edit as Needed)")
            chief_complaint = gr.Textbox(label="Chief Complaint", lines=2, interactive=True)
            duration = gr.Textbox(label="Duration", lines=1, interactive=True)
            severity = gr.Textbox(label="Severity", lines=1, interactive=True)
            history = gr.Textbox(label="Relevant History", lines=3, interactive=True)

        with gr.Column(scale=1):
            gr.Markdown("### Quick Keywords (Reference Only)")
            keywords_display = gr.Textbox(
                label="For scanning during review — read only",
                lines=16,
                interactive=False,
                placeholder="Keywords will appear here..."
            )

    # Step 3: Save
    gr.Markdown("## Step 3: Save to Patient History")
    save_btn = gr.Button("Save Patient Record to Database", variant="primary", size="lg")
    save_status = gr.Textbox(label="Save Status", interactive=False)

    # Bonus: View patient history
    gr.Markdown("## View Patient Clinical History")
    with gr.Row():
        history_btn = gr.Button("View History for This Patient ID")
    history_display = gr.Textbox(label="Past Visits", lines=8, interactive=False)

    # ------------------------------------------------------------------
    # Wire up the pipeline
    # ------------------------------------------------------------------

    transcribe_btn.click(
        fn=process_patient_intake,
        inputs=[patient_id, language, audio_input],
        outputs=[
            chief_complaint, duration, severity, history,
            keywords_display, transcript_state, keywords_state, process_status
        ]
    )

    save_btn.click(
        fn=handle_save_click,
        inputs=[
            patient_id, language, chief_complaint, duration, severity, history,
            transcript_state, keywords_state
        ],
        outputs=[save_status]
    )

    history_btn.click(
        fn=handle_view_history_click,
        inputs=[patient_id],
        outputs=[history_display]
    )

if __name__ == "__main__":
    # share=True gives a public https://xxxx.gradio.live link — needed since
    # this runs on Colab's server, not your own machine. Link lasts ~72 hours.
    demo.launch(share=True, debug=True)
