"""
NCAIR-DSA — Gradio entry point.

Pipeline: audio capture -> audio-based language detection (BEFORE ASR) ->
ASR -> N-ATLaS structuring (clinical note + possible recommendations) ->
keyword extraction -> role-gated review -> patient record save -> exports.

Role model:
  - NURSE view: capture only (patient ID, audio, language detection/confirm).
    No access to clinical note editing, recommendations, or exports.
  - DOCTOR view: full review (Clinical Note / Possible Recommendations /
    Keywords tabs), save, patient history lookup, CSV/TXT exports.

This mirrors how the tool would realistically be used in a clinic: a nurse
handles intake, a doctor handles clinical review — they are not the same
job and should not see the same screen.

NOTE: Google Drive must be mounted BEFORE running this script, from a
notebook cell — not from inside app.py. In your Colab notebook:
    from google.colab import drive
    drive.mount('/content/drive')
THEN run: !python app.py
"""
import gradio as gr
import json

from asr.transcribe import transcribe_audio
from nlp.structure_note import structure_note
from nlp.extract_keywords import extract_keywords
from nlp.audio_language_detect import detect_audio_language
from templates.clinical_note import (
    init_db,
    save_patient_record as db_save_patient_record,
    get_patient_history,
    export_all_records_to_csv,
)

# Ensure database tables exist on startup
init_db()

RECOMMENDATIONS_DISCLAIMER = (
    "⚠️ For doctor consideration only — NOT a diagnosis. "
    "This is a model-generated suggestion and may be incomplete or inaccurate. "
    "Clinical judgment should always take precedence.\n\n"
)

SEVERITY_COLORS = {
    "mild": "#2e7d32",
    "moderate": "#ef6c00",
    "severe": "#c62828",
}


def severity_badge_html(severity_text: str) -> str:
    if not severity_text:
        return ""
    lowered = severity_text.lower()
    color = "#666666"
    label = "Unspecified"
    for key, hex_color in SEVERITY_COLORS.items():
        if key in lowered:
            color = hex_color
            label = key.capitalize()
            break
    return (
        f'<div style="display:inline-block; padding:4px 14px; border-radius:14px; '
        f'background-color:{color}; color:white; font-weight:600; font-size:0.85em;">'
        f'{label}</div>'
    )


# ============================================================================
# Language Detection (runs on raw audio, BEFORE ASR)
# ============================================================================

def run_language_detection(audio_file, manual_language):
    """
    Called when audio is captured. Detects language directly from audio.
    If confident, suggests it (nurse can still override). If not confident,
    falls back to whatever the nurse manually selected.
    """
    if not audio_file:
        return manual_language, "No audio yet — select language manually or record audio first."

    result = detect_audio_language(audio_file)

    if result["auto_detect_reliable"]:
        detected = result["detected_language"]
        msg = f"✓ Auto-detected: {detected} (confidence: {result['confidence']:.0%})"
        return detected, msg
    else:
        msg = (
            f"⚠ Auto-detection not confident enough (top guess confidence: "
            f"{result['confidence']:.0%}). Please confirm the language manually."
        )
        return manual_language, msg


# ============================================================================
# Main Pipeline
# ============================================================================

def process_patient_intake(patient_id, language, audio_file):
    """
    1. Transcribe audio (language already decided by this point — either
       auto-detected or manually confirmed, upstream of this function)
    2. Structure into full English clinical note + possible recommendations
    3. Extract keywords from the ENGLISH note (not raw transcript)
    """
    if not audio_file or not patient_id:
        return "", "", "", "", "", "", "", "", "", "{}", "Error: Patient ID and audio required"

    try:
        transcript = transcribe_audio(audio_file, language)
    except Exception as e:
        return "", "", "", "", "", "", "", "", "", "{}", f"Error transcribing audio: {str(e)}"

    language_context = f"Language: {language} (confirmed prior to ASR via audio-based detection or nurse selection)."
    note = structure_note(transcript, language_context=language_context)

    if note.get("_error"):
        status = f"⚠ Note generated with issues: {note['_error']}"
    else:
        status = "✓ Ready for doctor review"

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

    severity_html = severity_badge_html(note.get("severity", ""))
    recommendations_with_disclaimer = RECOMMENDATIONS_DISCLAIMER + note.get("possible_recommendations", "")

    return (
        note.get("chief_complaint", ""),
        note.get("duration", ""),
        note.get("severity", ""),
        note.get("history", ""),
        severity_html,
        recommendations_with_disclaimer,
        keywords_text,
        transcript,             # shown separately, kept distinct from the structured note
        transcript,             # hidden state — needed at save time
        json.dumps(keywords),   # hidden state — needed at save time
        status,
    )


def handle_save_click(patient_id, language, chief_complaint, duration, severity, history,
                       recommendations_text, transcript_state, keywords_state_json):
    if not patient_id:
        return "Error: Patient ID is required to save"
    try:
        keywords = json.loads(keywords_state_json) if keywords_state_json else {}

        recommendations_to_save = recommendations_text
        if recommendations_to_save.startswith(RECOMMENDATIONS_DISCLAIMER):
            recommendations_to_save = recommendations_to_save[len(RECOMMENDATIONS_DISCLAIMER):]

        db_save_patient_record(
            patient_id=patient_id,
            chief_complaint=chief_complaint,
            duration=duration,
            severity=severity,
            history=history,
            possible_recommendations=recommendations_to_save,
            language=language,
            keywords=keywords,
            transcript=transcript_state,
        )
        return f"✓ Patient {patient_id} record saved successfully"
    except Exception as e:
        return f"✗ Error saving record: {str(e)}"


def handle_view_history_click(patient_id):
    if not patient_id:
        return "Enter a Patient ID to view history"
    visits = get_patient_history(patient_id)
    if not visits:
        return f"No previous visits found for patient {patient_id}"
    lines = [f"History for {patient_id} ({len(visits)} visit(s)):\n"]
    for visit in visits:
        visit_id, chief_complaint, duration, severity, history, recommendations, language, created_at = visit
        lines.append(
            f"— {created_at} [{language}]\n"
            f"  Complaint: {chief_complaint}\n"
            f"  Duration: {duration}\n"
            f"  Severity: {severity}\n"
            f"  History: {history}\n"
            f"  Recommendations: {recommendations}\n"
        )
    return "\n".join(lines)


# ============================================================================
# Exports
# ============================================================================

def handle_export_csv():
    """Export the full patient database as a downloadable CSV."""
    try:
        path, row_count = export_all_records_to_csv("patient_records_export.csv")
        return path, f"✓ Exported {row_count} record(s) to CSV"
    except Exception as e:
        return None, f"✗ Export failed: {str(e)}"


def handle_export_transcript_txt(patient_id, chief_complaint, duration, severity,
                                  history, transcript_state):
    """
    Export the translated (English) clinical note as plain text, alongside
    the original untranslated transcript for reference — so if N-ATLaS
    mistranslates something, the original is not lost.
    """
    if not patient_id:
        return None, "Error: Patient ID required to export"

    filename = f"transcript_{patient_id}.txt"
    content = f"""NCAIR-DSA — Patient Transcript Export
Patient ID: {patient_id}

=== TRANSLATED / STRUCTURED CLINICAL NOTE ===

Chief Complaint:
{chief_complaint}

Duration:
{duration}

Severity:
{severity}

Relevant History:
{history}

=== ORIGINAL UNTRANSLATED TRANSCRIPT ===
(kept separately in case of mistranslation or ASR error)

{transcript_state}
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)

    return filename, "✓ Transcript exported"


# ============================================================================
# Role Switching
# ============================================================================

def switch_role(role):
    """Toggle visibility of components based on selected role."""
    is_doctor = (role == "Doctor")
    return (
        gr.update(visible=is_doctor),  # doctor_review_group
        gr.update(visible=is_doctor),  # save_group
        gr.update(visible=is_doctor),  # history_group
        gr.update(visible=is_doctor),  # export_group
    )


# ============================================================================
# Theme
# ============================================================================

theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
)


# ============================================================================
# Gradio Interface
# ============================================================================

with gr.Blocks(title="NCAIR-DSA: Patient Intake Assistant", theme=theme) as demo:
    gr.HTML(
        """
        <div style="text-align:center; padding: 12px 0 4px 0;">
            <h1 style="margin-bottom:4px;">🩺 NCAIR-DSA Patient Intake Assistant</h1>
            <p style="color:#666; font-size:1.05em;">
                Multi-lingual voice-to-clinical-note system — Hausa · Igbo · Yoruba → English
            </p>
        </div>
        """
    )

    transcript_state = gr.State("")
    keywords_state = gr.State("{}")

    # --- Role Selection ---
    with gr.Group():
        gr.Markdown("### 👤 Select Your Role")
        role_selector = gr.Radio(
            choices=["Nurse", "Doctor"],
            value="Nurse",
            label="Who is using the system right now?",
            info="Nurses handle intake capture. Doctors review, edit, save, and export."
        )

    # --- Step 1: Capture (visible to both roles) ---
    with gr.Group():
        gr.Markdown("### 🎙️ Step 1 — Record Patient Audio")
        with gr.Row():
            patient_id = gr.Textbox(label="Patient ID", placeholder="e.g., P001", scale=1)
            language = gr.Dropdown(
                choices=["Hausa", "Igbo", "Yoruba"],
                label="Language (auto-detected — confirm or override)",
                scale=1
            )

        audio_input = gr.Audio(
            label="Patient Audio (Upload or Record)",
            type="filepath",
            sources=["upload", "microphone"]
        )

        detect_lang_btn = gr.Button("🌐 Detect Language from Audio", size="sm")
        language_detect_status = gr.Textbox(label="Language Detection Status", interactive=False)

        transcribe_btn = gr.Button("🔄 Transcribe & Generate Clinical Note", variant="primary", size="lg")
        process_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("**Original Transcript (untranslated, kept separately for reference):**")
        raw_transcript_display = gr.Textbox(label="Raw Transcript", lines=3, interactive=False)

    # --- Step 2: Doctor Review (Doctor role only) ---
    with gr.Group(visible=False) as doctor_review_group:
        gr.Markdown("### 📋 Step 2 — Doctor Review & Edit")

        with gr.Tabs():
            with gr.Tab("🩺 Clinical Note"):
                gr.Markdown("_Editable — grounded strictly in what the patient said._")
                chief_complaint = gr.Textbox(label="Chief Complaint", lines=4, interactive=True)
                duration = gr.Textbox(label="Duration", lines=3, interactive=True)
                with gr.Row():
                    severity = gr.Textbox(label="Severity", lines=3, interactive=True, scale=3)
                    severity_badge = gr.HTML(label="", scale=1)
                history = gr.Textbox(label="Relevant History", lines=5, interactive=True)

            with gr.Tab("💡 Possible Recommendations"):
                gr.Markdown(
                    "_AI-suggested considerations for triage — editable. "
                    "Not a diagnosis; hedged and conservative by design._"
                )
                possible_recommendations = gr.Textbox(
                    label="Possible Causes / Effects / Considerations",
                    lines=8,
                    interactive=True
                )

            with gr.Tab("🔑 Keywords (reference)"):
                gr.Markdown("_Quick-scan reference only — not part of the saved clinical note._")
                keywords_display = gr.Textbox(label="Extracted keywords", lines=12, interactive=False)

    # --- Step 3: Save (Doctor role only) ---
    with gr.Group(visible=False) as save_group:
        gr.Markdown("### 💾 Step 3 — Save to Patient History")
        save_btn = gr.Button("Save Patient Record to Database", variant="primary", size="lg")
        save_status = gr.Textbox(label="Save Status", interactive=False)

    # --- History (Doctor role only) ---
    with gr.Group(visible=False) as history_group:
        gr.Markdown("### 📖 View Patient Clinical History")
        history_btn = gr.Button("View History for This Patient ID")
        history_display = gr.Textbox(label="Past Visits", lines=8, interactive=False)

    # --- Exports (Doctor role only) ---
    with gr.Group(visible=False) as export_group:
        gr.Markdown("### 📤 Export")
        with gr.Row():
            with gr.Column():
                export_csv_btn = gr.Button("Export Full Database as CSV")
                export_csv_file = gr.File(label="Database CSV", interactive=False)
                export_csv_status = gr.Textbox(label="Export Status", interactive=False)
            with gr.Column():
                export_txt_btn = gr.Button("Export This Transcript as Plain Text")
                export_txt_file = gr.File(label="Transcript TXT", interactive=False)
                export_txt_status = gr.Textbox(label="Export Status", interactive=False)

    # ------------------------------------------------------------------
    # Wire up events
    # ------------------------------------------------------------------

    role_selector.change(
        fn=switch_role,
        inputs=[role_selector],
        outputs=[doctor_review_group, save_group, history_group, export_group]
    )

    detect_lang_btn.click(
        fn=run_language_detection,
        inputs=[audio_input, language],
        outputs=[language, language_detect_status]
    )

    transcribe_btn.click(
        fn=process_patient_intake,
        inputs=[patient_id, language, audio_input],
        outputs=[
            chief_complaint, duration, severity, history, severity_badge,
            possible_recommendations, keywords_display,
            raw_transcript_display, transcript_state, keywords_state, process_status
        ]
    )

    save_btn.click(
        fn=handle_save_click,
        inputs=[
            patient_id, language, chief_complaint, duration, severity, history,
            possible_recommendations, transcript_state, keywords_state
        ],
        outputs=[save_status]
    )

    history_btn.click(
        fn=handle_view_history_click,
        inputs=[patient_id],
        outputs=[history_display]
    )

    export_csv_btn.click(
        fn=handle_export_csv,
        inputs=[],
        outputs=[export_csv_file, export_csv_status]
    )

    export_txt_btn.click(
        fn=handle_export_transcript_txt,
        inputs=[patient_id, chief_complaint, duration, severity, history, transcript_state],
        outputs=[export_txt_file, export_txt_status]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
