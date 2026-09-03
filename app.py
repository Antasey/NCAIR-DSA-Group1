"""
MediVoice — Multi-Lingual Patient Intake Assistant (Gradio / hosted version)

Same backend logic and screen flow as the CustomTkinter desktop build
(gui/app.py, gui/nurse_flow.py, gui/doctor_flow.py) — restyled to run as a
hosted Gradio app instead of a native window, so nothing has to be
downloaded/installed by end users. Visual design matches the MediVoice
reference screenshot: white canvas, deep teal accent, rounded cards.

Screen flow (mirrors the desktop app):
    Dashboard (landing) --"New Intake"--> role picker
        --Nurse--> Nurse capture flow
        --Doctor--> Patients queue (same destination as the sidebar link)
    Patients (sidebar) --"Review"--> Doctor review screen --"Confirm & Finalize"--> back to queue

Run with: python app.py
Expects nlp/, asr/, templates/ as sibling folders (same layout as the rest
of this project) — nothing here changes those modules.
"""

import json
import os
import sys
import tempfile

import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nlp"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "asr"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "templates"))

import clinical_note as db  # noqa: E402

LANGUAGES = ["Hausa", "Igbo", "Yoruba"]

RECOMMENDATIONS_DISCLAIMER = (
    "⚠️ For doctor consideration only — NOT a diagnosis. This is a "
    "model-generated suggestion and may be incomplete or inaccurate. "
    "Clinical judgment should always take precedence."
)

# ============================================================================
# THEME — ported from gui/theme.py
# ============================================================================

COLOR_BG = "#F7F9FA"
COLOR_WHITE = "#FFFFFF"
COLOR_TEAL = "#1B6B6B"
COLOR_TEAL_HOVER = "#155454"
COLOR_TEAL_LIGHT = "#E4F0EF"
COLOR_TEXT_DARK = "#16232E"
COLOR_TEXT_BODY = "#374151"
COLOR_TEXT_MUTED = "#6B7280"
COLOR_CARD_BORDER = "#E5E7EB"
COLOR_SUCCESS = "#22C55E"
COLOR_WARNING = "#F59E0B"
COLOR_DANGER = "#DC2626"

SEVERITY_COLORS = {"mild": "#2E7D32", "moderate": "#EF6C00", "severe": "#C62828"}

CUSTOM_CSS = f"""
.gradio-container {{ background: {COLOR_BG} !important; font-family: 'Segoe UI', sans-serif; }}
#sidebar {{ background: {COLOR_WHITE}; border-right: 1px solid {COLOR_CARD_BORDER}; min-height: 100vh; }}
.nav-btn {{ text-align: left !important; justify-content: flex-start !important; }}
.nav-btn.active {{ background: {COLOR_TEAL} !important; color: {COLOR_WHITE} !important; }}
.card {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px; padding: 20px; }}
.stat-card {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
              padding: 20px; }}
.stat-label {{ color: {COLOR_TEXT_MUTED}; font-size: 11px; font-weight: 700; letter-spacing: .04em; }}
.stat-number {{ color: {COLOR_TEXT_DARK}; font-size: 32px; font-weight: 700; margin: 4px 0; }}
.stat-caption {{ color: {COLOR_TEXT_MUTED}; font-size: 12px; }}
.cta-banner {{ background: linear-gradient(135deg, {COLOR_TEAL}, {COLOR_TEAL_HOVER}); border-radius: 14px;
               padding: 28px 32px; color: {COLOR_WHITE}; }}
.cta-eyebrow {{ font-size: 11px; font-weight: 700; letter-spacing: .04em; color: {COLOR_TEAL_LIGHT}; }}
.cta-title {{ font-size: 26px; font-weight: 700; margin: 6px 0 8px; }}
.cta-body {{ color: {COLOR_TEAL_LIGHT}; font-size: 13px; }}
.disclaimer {{ color: {COLOR_WARNING}; font-size: 12px; }}
.status-ok {{ color: {COLOR_SUCCESS}; }}
.status-err {{ color: {COLOR_DANGER}; }}
.queue-row {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
              padding: 14px 20px; margin-bottom: 8px; }}
#primary-btn {{ background: {COLOR_TEAL} !important; color: {COLOR_WHITE} !important; border: none !important; }}
#primary-btn:hover {{ background: {COLOR_TEAL_HOVER} !important; }}
"""

# ============================================================================
# HELPERS — building the same HTML blocks the desktop app draws with CTk
# ============================================================================


def render_dashboard():
    stats = db.get_dashboard_stats()
    stat_html = f"""
    <div style="display:flex; gap:16px; margin-bottom:20px;">
      <div class="stat-card" style="flex:1;">
        <div class="stat-label">TOTAL PATIENTS</div>
        <div class="stat-number">{stats['total_patients']}</div>
        <div class="stat-caption">Registered patients</div>
      </div>
      <div class="stat-card" style="flex:1;">
        <div class="stat-label">CLINICAL VISITS</div>
        <div class="stat-number">{stats['total_visits']}</div>
        <div class="stat-caption">Recorded visits</div>
      </div>
      <div class="stat-card" style="flex:1;">
        <div class="stat-label">PENDING REVIEW</div>
        <div class="stat-number">{stats['pending_review']}</div>
        <div class="stat-caption">Awaiting doctor</div>
      </div>
    </div>
    <div class="cta-banner">
      <div class="cta-eyebrow">START A NEW PATIENT INTAKE</div>
      <div class="cta-title">Capture patient information<br/>through voice-powered intake.</div>
      <div class="cta-body">Record symptoms in Igbo, Hausa or Yoruba and transform the
      conversation into structured clinical information.</div>
    </div>
    """
    pending = db.get_pending_visits()
    if not pending:
        activity_html = f'<p style="color:{COLOR_TEXT_MUTED};">No recent activity yet.</p>'
    else:
        rows = ""
        for visit_id, patient_id, chief_complaint, language, created_at in pending[:5]:
            rows += f"""
            <div class="queue-row">
              <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{patient_id} · {chief_complaint or 'Pending structuring'}</div>
              <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at} · Pending review</div>
            </div>
            """
        activity_html = rows
    return stat_html, activity_html


def render_queue():
    pending = db.get_pending_visits()
    if not pending:
        return f'<div class="card"><span class="status-ok">✓ No visits waiting for review.</span></div>', []
    rows_html = ""
    choices = []
    for visit_id, patient_id, chief_complaint, language, created_at in pending:
        rows_html += f"""
        <div class="queue-row">
          <div style="font-weight:700; color:{COLOR_TEXT_DARK};">Patient {patient_id}</div>
          <div style="color:{COLOR_TEXT_BODY};">{chief_complaint or 'Awaiting structuring'}</div>
          <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at}</div>
        </div>
        """
        choices.append(f"Visit #{visit_id} — {patient_id}")
    return rows_html, choices


def severity_color(severity_text):
    if not severity_text:
        return COLOR_TEXT_MUTED
    lowered = severity_text.lower()
    for key, color in SEVERITY_COLORS.items():
        if key in lowered:
            return color
    return COLOR_TEXT_MUTED


def format_keywords(keywords_json):
    try:
        data = json.loads(keywords_json) if keywords_json else {}
    except json.JSONDecodeError:
        data = {}
    lines = []
    for category in ["symptoms", "duration", "severity", "anatomical_sites"]:
        values = data.get(category, [])
        lines.append(f"{category.replace('_', ' ').title()}: " + (", ".join(values) if values else "None"))
    return "\n".join(lines)


def extract_visit_id(choice_label):
    # "Visit #12 — P001" -> 12
    return int(choice_label.split("—")[0].replace("Visit #", "").strip())


# ============================================================================
# BACKEND ACTIONS — identical calls to nurse_flow.py / doctor_flow.py
# ============================================================================


def run_language_detection(audio_path):
    if not audio_path:
        return gr.update(), ""
    try:
        from audio_language_detect import detect_audio_language
        result = detect_audio_language(audio_path)
        if result["auto_detect_reliable"]:
            msg = f"✓ Auto-detected: {result['detected_language']} (confidence: {result['confidence']:.0%})"
            lang_update = gr.update(value=result["detected_language"])
        else:
            msg = f"⚠ Detection not confident ({result['confidence']:.0%}) — confirm language manually."
            lang_update = gr.update()
        if result.get("is_code_switched"):
            candidates = ", ".join(f"{l} ({p:.0%})" for l, p in result["code_switch_candidates"])
            msg += f"\n⚠ Possible code-switching detected: {candidates}. Processing will continue."
        return lang_update, msg
    except Exception as e:
        return gr.update(), f"⚠ Language detection unavailable: {e}. Select language manually."


def process_intake(patient_id, language, audio_path):
    if not patient_id or not patient_id.strip():
        return "✗ Please enter a Patient ID before processing.", *render_dashboard()
    if not audio_path:
        return "✗ Please provide patient audio before processing.", *render_dashboard()

    try:
        from transcribe import transcribe_audio
        from structure_note import structure_note
        from extract_keywords import extract_keywords
        from audio_language_detect import detect_audio_language

        language_context = ""
        try:
            detection = detect_audio_language(audio_path)
            language_context = detection.get("context_note", "")
        except Exception:
            pass  # non-fatal — proceed without extra context

        transcript = transcribe_audio(audio_path, language)
        note = structure_note(transcript, language_context=language_context)
        keywords = extract_keywords(transcript)

        visit_id = db.save_patient_record(
            patient_id=patient_id.strip(),
            chief_complaint=note.get("chief_complaint", ""),
            duration=note.get("duration", ""),
            severity=note.get("severity", ""),
            history=note.get("history", ""),
            possible_recommendations=note.get("possible_recommendations", ""),
            language=language,
            keywords=keywords,
            transcript=transcript,
        )
        status = f"✓ Saved. Patient {patient_id}'s visit (#{visit_id}) is now awaiting doctor review."
    except Exception as e:
        status = f"✗ Error: {e}"

    stat_html, activity_html = render_dashboard()
    return status, stat_html, activity_html


def load_review(choice_label):
    if not choice_label:
        return ("", "", "", "", "", "", "", "", gr.update(), None)
    visit_id = extract_visit_id(choice_label)
    visit = db.get_visit_by_id(visit_id)
    if not visit:
        return ("Visit not found.", "", "", "", "", "", "", "", gr.update(), None)

    (v_id, patient_id, chief_complaint, duration, severity, history,
     recommendations, language, keywords_json, transcript, status, created_at) = visit

    header = f"Patient {patient_id} · {language} · Visit #{v_id} · {created_at}"
    return (
        header, chief_complaint or "", duration or "", severity or "",
        history or "", recommendations or "", format_keywords(keywords_json),
        transcript or "", gr.update(), visit_id,
    )


def confirm_and_finalize(visit_id, chief_complaint, duration, severity, history, recommendations):
    if visit_id is None:
        return "✗ No visit selected.", *render_queue()
    db.update_visit_and_mark_reviewed(visit_id, chief_complaint, duration, severity, history, recommendations)
    msg = f"✓ Visit #{visit_id} finalized."
    rows_html, choices = render_queue()
    return msg, rows_html, choices


def export_csv(visit_id):
    if visit_id is None:
        return None
    tmp_path = os.path.join(tempfile.gettempdir(), f"visit_{visit_id}.csv")
    db.export_single_visit_to_csv(visit_id, tmp_path)
    return tmp_path


# ============================================================================
# APP LAYOUT
# ============================================================================

db.init_db()

with gr.Blocks(css=CUSTOM_CSS, title="MediVoice — Multi-Lingual Patient Intake Assistant") as demo:
    visit_id_state = gr.State(None)

    with gr.Row():
        # ---------------- Sidebar ----------------
        with gr.Column(scale=1, elem_id="sidebar", min_width=240):
            gr.HTML(f"""
                <div style="display:flex; align-items:center; gap:12px; padding:20px 8px;">
                  <div style="width:44px;height:44px;border-radius:12px;background:{COLOR_TEAL};
                              display:flex;align-items:center;justify-content:center;
                              color:white;font-size:22px;font-weight:700;">+</div>
                  <div>
                    <div style="font-weight:700; color:{COLOR_TEXT_DARK}; font-size:16px;">MediVoice</div>
                    <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Patient Intake</div>
                  </div>
                </div>
                <div style="color:{COLOR_TEXT_MUTED}; font-size:11px; font-weight:700;
                            padding:8px 8px 4px;">WORKSPACE</div>
            """)
            nav_dashboard = gr.Button("🏠  Dashboard", elem_classes=["nav-btn", "active"])
            nav_patients = gr.Button("🧑‍⚕️  Patients", elem_classes=["nav-btn"])
            nav_new_intake = gr.Button("🎙️  New Intake", elem_classes=["nav-btn"])
            gr.HTML(f"""
                <div style="margin-top:40px; background:{COLOR_TEAL_LIGHT}; border-radius:14px; padding:12px 14px;">
                  <div style="color:{COLOR_SUCCESS};">● <b style="color:{COLOR_TEXT_DARK};">System operational</b></div>
                  <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Speech services ready</div>
                </div>
            """)

        # ---------------- Main content ----------------
        with gr.Column(scale=4):

            # ---- Dashboard page ----
            with gr.Column(visible=True) as dashboard_page:
                gr.HTML(f'<div style="color:{COLOR_TEXT_MUTED};">Good day 👋</div>'
                        f'<div style="font-size:28px; font-weight:700; color:{COLOR_TEXT_DARK};">Patient Intake Dashboard</div>'
                        f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Manage patient intake and clinical information efficiently.</div>')
                dash_stats_html = gr.HTML()
                start_intake_btn = gr.Button("Start Intake  →", elem_id="primary-btn")
                gr.HTML(f'<div style="font-size:18px; font-weight:700; color:{COLOR_TEXT_DARK}; margin:20px 0 8px;">Recent activity</div>')
                dash_activity_html = gr.HTML()

            # ---- Role picker page (matches the desktop app's modal) ----
            with gr.Column(visible=False) as role_page:
                back_from_role = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(f'<div style="text-align:center; margin-top:20px;">'
                        f'<div style="font-size:22px; font-weight:700; color:{COLOR_TEXT_DARK};">Who\'s using the system?</div>'
                        f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Select a role to continue</div></div>')
                with gr.Row():
                    role_nurse_btn = gr.Button("🧑‍⚕️  Nurse\nRecord patient intake", elem_classes=["card"])
                    role_doctor_btn = gr.Button("🩺  Doctor\nReview pending visits", elem_classes=["card"])

            # ---- Nurse intake page ----
            with gr.Column(visible=False) as intake_page:
                back_from_intake = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">New Patient Intake</div>'
                        f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Record or upload patient audio to begin.</div>')
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Details</b>')
                    with gr.Row():
                        patient_id_input = gr.Textbox(label="Patient ID", placeholder="e.g. P001")
                        language_input = gr.Dropdown(LANGUAGES, value=LANGUAGES[0],
                                                      label="Language (auto-detected — confirm or override)")
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Audio</b>')
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload")
                    lang_detect_status = gr.Markdown("")
                process_btn = gr.Button("🔄  Transcribe & Structure Note", elem_id="primary-btn")
                intake_status = gr.Markdown("")

            # ---- Doctor queue page ----
            with gr.Column(visible=False) as queue_page:
                back_from_queue = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">Patients</div>'
                        f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Visits awaiting doctor review, oldest first.</div>')
                queue_html = gr.HTML()
                queue_select = gr.Dropdown(label="Select a visit to review", choices=[])
                review_btn = gr.Button("Review →", elem_id="primary-btn")

            # ---- Doctor review page ----
            with gr.Column(visible=False) as review_page:
                back_from_review = gr.Button("← Back to Patients", elem_classes=["nav-btn"])
                review_header = gr.Markdown("")
                with gr.Tabs():
                    with gr.Tab("🩺 Clinical Note"):
                        gr.Markdown("Editable — grounded strictly in what the patient said.")
                        chief_complaint_box = gr.Textbox(label="Chief Complaint", lines=3)
                        duration_box = gr.Textbox(label="Duration", lines=2)
                        severity_box = gr.Textbox(label="Severity", lines=2)
                        history_box = gr.Textbox(label="Relevant History", lines=4)
                    with gr.Tab("💡 Possible Recommendations"):
                        gr.HTML(f'<div class="disclaimer">{RECOMMENDATIONS_DISCLAIMER}</div>')
                        recommendations_box = gr.Textbox(label="", lines=6)
                    with gr.Tab("🔑 Keywords"):
                        gr.Markdown("Quick-scan reference only — not part of the saved clinical note.")
                        keywords_display = gr.Textbox(label="", lines=6, interactive=False)
                gr.Markdown("**Original Transcript** (untranslated, kept for reference)")
                transcript_display = gr.Textbox(label="", lines=3, interactive=False)
                with gr.Row():
                    finalize_btn = gr.Button("✓  Confirm & Finalize", elem_id="primary-btn")
                    export_btn = gr.Button("📤  Export Visit as CSV")
                review_status = gr.Markdown("")
                export_file = gr.File(label="Download", visible=True)

    # ------------------------------------------------------------------
    # Navigation wiring
    # ------------------------------------------------------------------

    all_pages = [dashboard_page, role_page, intake_page, queue_page, review_page]

    def goto(page_index, *_):
        return [gr.update(visible=(i == page_index)) for i in range(len(all_pages))]

    def refresh_dashboard():
        stat_html, activity_html = render_dashboard()
        return stat_html, activity_html

    def refresh_queue():
        rows_html, choices = render_queue()
        return rows_html, gr.update(choices=choices, value=None)

    nav_dashboard.click(lambda: goto(0), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    nav_new_intake.click(lambda: goto(1), outputs=all_pages)  # -> role picker, same as desktop app's modal
    start_intake_btn.click(lambda: goto(1), outputs=all_pages)  # -> role picker
    nav_patients.click(lambda: goto(3), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    # Role picker: Nurse -> intake flow, Doctor -> same Patients queue as the sidebar link
    # (mirrors the desktop app's _select_role, which routes Doctor to show_doctor_queue()
    # rather than dead-ending)
    role_nurse_btn.click(lambda: goto(2), outputs=all_pages)
    role_doctor_btn.click(lambda: goto(3), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])
    back_from_role.click(lambda: goto(0), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])

    back_from_intake.click(lambda: goto(0), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_queue.click(lambda: goto(0), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_review.click(lambda: goto(3), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    # ------------------------------------------------------------------
    # Nurse intake wiring
    # ------------------------------------------------------------------

    audio_input.change(run_language_detection, inputs=[audio_input], outputs=[language_input, lang_detect_status])
    process_btn.click(
        process_intake,
        inputs=[patient_id_input, language_input, audio_input],
        outputs=[intake_status, dash_stats_html, dash_activity_html],
    )

    # ------------------------------------------------------------------
    # Doctor queue / review wiring
    # ------------------------------------------------------------------

    review_btn.click(lambda: goto(4), outputs=all_pages).then(
        load_review,
        inputs=[queue_select],
        outputs=[review_header, chief_complaint_box, duration_box, severity_box, history_box,
                 recommendations_box, keywords_display, transcript_display, review_status, visit_id_state],
    )

    finalize_btn.click(
        confirm_and_finalize,
        inputs=[visit_id_state, chief_complaint_box, duration_box, severity_box, history_box, recommendations_box],
        outputs=[review_status, queue_html, queue_select],
    )

    export_btn.click(export_csv, inputs=[visit_id_state], outputs=[export_file])


if __name__ == "__main__":
    demo.launch()
