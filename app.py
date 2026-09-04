"""
MediVoice — Multi-Lingual Patient Intake Assistant (Gradio / hosted version)

With:
  • Loading/processing indicators on all heavy operations
  • Confirmation messages after save with "Return to Dashboard" action
  • Back-to-Dashboard on every page
  • Transformers-based N-ATLaS (run `python setup_models.py` first)
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

# ── pre-flight model check ──────────────────────────────────────────────────
MODEL_DIR = Path("models/N-ATLaS")
if not MODEL_DIR.exists():
    raise RuntimeError(
        f"N-ATLaS model not found at {MODEL_DIR}.\n"
        f"Please run first:  python setup_models.py"
    )

# ── path setup ──────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ("nlp", "asr", "templates"):
    p = os.path.join(_BASE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import clinical_note as db  # noqa: E402

# ── logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────────────
LANGUAGES = ["Hausa", "Igbo", "Yoruba"]

RECOMMENDATIONS_DISCLAIMER = (
    "⚠️ For doctor consideration only — NOT a diagnosis. This is a "
    "model-generated suggestion and may be incomplete or inaccurate. "
    "Clinical judgment should always take precedence."
)

# ── theme ───────────────────────────────────────────────────────────────────
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
.status-warn {{ color: {COLOR_WARNING}; }}
.queue-row {{ background: {COLOR_WHITE}; border: 1px solid {COLOR_CARD_BORDER}; border-radius: 14px;
              padding: 14px 20px; margin-bottom: 8px; }}
#primary-btn {{ background: {COLOR_TEAL} !important; color: {COLOR_WHITE} !important; border: none !important; }}
#primary-btn:hover {{ background: {COLOR_TEAL_HOVER} !important; }}
#primary-btn:disabled {{ background: #9CA3AF !important; cursor: not-allowed !important; }}
.preview-card {{ background: {COLOR_TEAL_LIGHT}; border: 1px solid {COLOR_TEAL}; border-radius: 14px; padding: 16px; margin-top: 12px; }}
.confirm-banner {{ background: {COLOR_TEAL_LIGHT}; border-left: 4px solid {COLOR_TEAL}; border-radius: 8px; padding: 12px 16px; margin: 12px 0; }}
.processing-spinner {{ color: {COLOR_TEAL}; font-size: 14px; font-weight: 600; }}
"""

# ── simple TTL cache ──────────────────────────────────────────────────────
class _TTLCache:
    def __init__(self, ttl_seconds: float = 5.0):
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def get(self, key: str) -> Any | None:
        with self._lock:
            val, ts = self._store.get(key, (None, 0))
            if val is not None and (time.time() - ts) < self._ttl:
                return val
            return None

    def set(self, key: str, val: Any) -> None:
        with self._lock:
            self._store[key] = (val, time.time())

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._store.pop(key, None)

    def invalidate_all(self) -> None:
        with self._lock:
            self._store.clear()


_cache = _TTLCache(ttl_seconds=5.0)

# ── data classes ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DetectionResult:
    detected_language: str
    confidence: float
    reliable: bool
    is_code_switched: bool
    code_switch_candidates: list[tuple[str, float]]
    context_note: str = ""


# ── HTML helpers ────────────────────────────────────────────────────────────
def _stat_card(label: str, number: int | str, caption: str) -> str:
    return f"""<div class="stat-card" style="flex:1;">
      <div class="stat-label">{label}</div>
      <div class="stat-number">{number}</div>
      <div class="stat-caption">{caption}</div>
    </div>"""


def _queue_row(patient_id: str, chief: str | None, language: str, created_at: str) -> str:
    return f"""<div class="queue-row">
      <div style="font-weight:700; color:{COLOR_TEXT_DARK};">{patient_id} · {chief or "Pending structuring"}</div>
      <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at} · Pending review</div>
    </div>"""


def _sidebar_logo() -> str:
    return f"""<div style="display:flex; align-items:center; gap:12px; padding:20px 8px;">
      <div style="width:44px;height:44px;border-radius:12px;background:{COLOR_TEAL};
                  display:flex;align-items:center;justify-content:center;
                  color:white;font-size:22px;font-weight:700;">+</div>
      <div>
        <div style="font-weight:700; color:{COLOR_TEXT_DARK}; font-size:16px;">MediVoice</div>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Patient Intake</div>
      </div>
    </div>
    <div style="color:{COLOR_TEXT_MUTED}; font-size:11px; font-weight:700;
                padding:8px 8px 4px;">WORKSPACE</div>"""


def _system_status() -> str:
    return f"""<div style="margin-top:40px; background:{COLOR_TEAL_LIGHT}; border-radius:14px; padding:12px 14px;">
      <div style="color:{COLOR_SUCCESS};">● <b style="color:{COLOR_TEXT_DARK};">System operational</b></div>
      <div style="color:{COLOR_TEXT_MUTED}; font-size:11px;">Speech services ready</div>
    </div>"""


# ── rendering ───────────────────────────────────────────────────────────────
def render_dashboard() -> tuple[str, str]:
    cached = _cache.get("dashboard")
    if cached is not None:
        return cached

    try:
        stats = db.get_dashboard_stats()
    except Exception:
        stats = {"total_patients": 0, "total_visits": 0, "pending_review": 0}

    stat_html = f"""<div style="display:flex; gap:16px; margin-bottom:20px;">
      {_stat_card("TOTAL PATIENTS", stats.get("total_patients", 0), "Registered patients")}
      {_stat_card("CLINICAL VISITS", stats.get("total_visits", 0), "Recorded visits")}
      {_stat_card("PENDING REVIEW", stats.get("pending_review", 0), "Awaiting doctor")}
    </div>
    <div class="cta-banner">
      <div class="cta-eyebrow">START A NEW PATIENT INTAKE</div>
      <div class="cta-title">Capture patient information<br/>through voice-powered intake.</div>
      <div class="cta-body">Record symptoms in Igbo, Hausa or Yoruba and transform the
      conversation into structured clinical information.</div>
    </div>"""

    try:
        pending = db.get_pending_visits()
    except Exception:
        pending = []

    if not pending:
        activity_html = f'<p style="color:{COLOR_TEXT_MUTED};">No recent activity yet.</p>'
    else:
        rows = "".join(
            _queue_row(pid, chief, lang, created)
            for vid, pid, chief, lang, created in pending[:5]
        )
        activity_html = rows

    result = (stat_html, activity_html)
    _cache.set("dashboard", result)
    return result


def render_queue() -> tuple[str, list[str]]:
    cached = _cache.get("queue")
    if cached is not None:
        return cached

    try:
        pending = db.get_pending_visits()
    except Exception:
        pending = []

    if not pending:
        result = (
            f'<div class="card"><span class="status-ok">✓ No visits waiting for review.</span></div>',
            [],
        )
        _cache.set("queue", result)
        return result

    rows_html = ""
    choices: list[str] = []
    for visit_id, patient_id, chief_complaint, language, created_at in pending:
        rows_html += f"""<div class="queue-row">
          <div style="font-weight:700; color:{COLOR_TEXT_DARK};">Patient {patient_id}</div>
          <div style="color:{COLOR_TEXT_BODY};">{chief_complaint or "Awaiting structuring"}</div>
          <div style="color:{COLOR_TEXT_MUTED}; font-size:12px;">{language} · {created_at}</div>
        </div>"""
        choices.append(f"Visit #{visit_id} — {patient_id}")

    result = (rows_html, choices)
    _cache.set("queue", result)
    return result


# ── formatting helpers ──────────────────────────────────────────────────────
def severity_color(severity_text: str | None) -> str:
    if not severity_text:
        return COLOR_TEXT_MUTED
    lowered = severity_text.lower()
    for key, color in SEVERITY_COLORS.items():
        if key in lowered:
            return color
    return COLOR_TEXT_MUTED


def format_keywords(keywords_json: str | None) -> str:
    try:
        data = json.loads(keywords_json) if keywords_json else {}
    except json.JSONDecodeError:
        data = {}

    lines = []
    for category in ("symptoms", "duration", "severity", "anatomical_sites"):
        values = data.get(category, [])
        label = category.replace("_", " ").title()
        lines.append(f"{label}: {', '.join(values) if values else 'None'}")
    return "\n".join(lines)


def extract_visit_id(choice_label: str | None) -> int | None:
    if not choice_label:
        return None
    m = re.search(r"Visit #\s*(\d+)", choice_label)
    return int(m.group(1)) if m else None


def _validate_patient_id(pid: str | None) -> tuple[bool, str]:
    if not pid or not pid.strip():
        return False, "✗ Please enter a Patient ID before processing."
    pid_clean = pid.strip()
    if len(pid_clean) > 50:
        return False, "✗ Patient ID is too long (max 50 characters)."
    if not re.match(r"^[A-Za-z0-9\-_]+$", pid_clean):
        return False, "✗ Patient ID may only contain letters, numbers, hyphens and underscores."
    return True, pid_clean


def _note_to_text(note: dict) -> str:
    parts = [
        note.get("chief_complaint", ""),
        note.get("duration", ""),
        note.get("severity", ""),
        note.get("history", ""),
    ]
    return "\n".join(p for p in parts if p)


def _format_nurse_preview(note: dict, transcript: str) -> str:
    chief = note.get("chief_complaint", "") or "Not mentioned"
    duration = note.get("duration", "") or "Not mentioned"
    severity = note.get("severity", "") or "Not mentioned"
    history = note.get("history", "") or "Not mentioned"

    return f"""<div class="preview-card">
      <div style="font-weight:700; color:{COLOR_TEAL}; font-size:14px; margin-bottom:10px;">📝 English Clinical Note Preview</div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Chief Complaint:</b> <span style="color:{COLOR_TEXT_BODY};">{chief}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Duration:</b> <span style="color:{COLOR_TEXT_BODY};">{duration}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">Severity:</b> <span style="color:{COLOR_TEXT_BODY};">{severity}</span></div>
      <div style="margin-bottom:8px;"><b style="color:{COLOR_TEXT_DARK};">History:</b> <span style="color:{COLOR_TEXT_BODY};">{history}</span></div>
      <div style="margin-top:10px; padding-top:10px; border-top:1px dashed {COLOR_CARD_BORDER};">
        <b style="color:{COLOR_TEXT_DARK};">Original Transcript:</b>
        <div style="color:{COLOR_TEXT_MUTED}; font-size:12px; margin-top:4px; font-style:italic;">{transcript or "No transcript available."}</div>
      </div>
    </div>"""


def _format_confirm_banner(patient_id: str, visit_id: int) -> str:
    return f"""<div class="confirm-banner">
      <div style="color:{COLOR_TEAL}; font-weight:700; font-size:14px;">✓ Visit Saved Successfully</div>
      <div style="color:{COLOR_TEXT_BODY}; font-size:13px; margin-top:4px;">
        Patient <b>{patient_id}</b>'s visit (#{visit_id}) has been saved and is now awaiting doctor review.
        Would you like to return to the Dashboard?
      </div>
    </div>"""


# ── backend actions ──────────────────────────────────────────────────────────
def run_language_detection(audio_path: str | None) -> tuple[gr.update, str, DetectionResult | None]:
    if not audio_path:
        return gr.update(), "", None

    try:
        from audio_language_detect import detect_audio_language
        result = detect_audio_language(audio_path)
    except Exception as e:
        return gr.update(), f"⚠ Language detection unavailable: {e}. Select language manually.", None

    dr = DetectionResult(
        detected_language=result.get("detected_language", LANGUAGES[0]),
        confidence=result.get("confidence", 0.0),
        reliable=result.get("auto_detect_reliable", False),
        is_code_switched=result.get("is_code_switched", False),
        code_switch_candidates=result.get("code_switch_candidates", []),
        context_note=result.get("context_note", ""),
    )

    if dr.reliable:
        msg = f"✓ Auto-detected: {dr.detected_language} (confidence: {dr.confidence:.0%})"
        lang_update = gr.update(value=dr.detected_language)
    else:
        msg = f"⚠ Detection not confident ({dr.confidence:.0%}) — confirm language manually."
        lang_update = gr.update()

    if dr.is_code_switched and dr.code_switch_candidates:
        candidates = ", ".join(f"{lang} ({prob:.0%})" for lang, prob in dr.code_switch_candidates)
        msg += f"\n⚠ Possible code-switching detected: {candidates}. Processing will continue."

    return lang_update, msg, dr


def process_intake(
    patient_id: str,
    language: str,
    audio_path: str | None,
    detection_result: DetectionResult | None,
) -> tuple:
    """
    Returns:
        (status, dash_stats, dash_activity, preview_visible, preview_html,
         chief, duration, severity, history, transcript, confirm_visible, confirm_html)
    """
    valid, msg_or_pid = _validate_patient_id(patient_id)
    if not valid:
        return msg_or_pid, *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), ""

    if not audio_path:
        return "✗ Please provide patient audio before processing.", *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), ""

    if language not in LANGUAGES:
        return f"✗ Unsupported language: {language}.", *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), ""

    try:
        from transcribe import transcribe_audio
        from structure_note import structure_note
        from extract_keywords import extract_keywords
        from audio_language_detect import detect_audio_language

        transcript = transcribe_audio(audio_path, language)
        if not transcript or not transcript.strip():
            return (
                "✗ Transcription returned empty. Please check audio quality and try again.",
                *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), "",
            )

        language_context = detection_result.context_note if detection_result else ""
        try:
            note = structure_note(transcript, language_context=language_context)
        except TypeError:
            note = structure_note(transcript)

        if "_error" in note:
            return (
                f"✗ Clinical note generation failed: {note['_error']}",
                *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), "",
            )

        if not note.get("chief_complaint"):
            return (
                "✗ The AI returned an empty clinical note. Please try again with clearer audio.",
                *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), "",
            )

        note_text = _note_to_text(note)
        keywords = extract_keywords(note_text)

        visit_id = db.save_patient_record(
            patient_id=msg_or_pid,
            chief_complaint=note.get("chief_complaint", ""),
            duration=note.get("duration", ""),
            severity=note.get("severity", ""),
            history=note.get("history", ""),
            possible_recommendations=note.get("possible_recommendations", ""),
            language=language,
            keywords=keywords,
            transcript=transcript,
        )

        status = f"✓ Saved. Patient {msg_or_pid}'s visit (#{visit_id}) is now awaiting doctor review."
        preview_html = _format_nurse_preview(note, transcript)
        confirm_html = _format_confirm_banner(msg_or_pid, visit_id)

        _cache.invalidate("dashboard")
        _cache.invalidate("queue")

    except Exception as e:
        traceback.print_exc()
        logger.error("Intake processing failed: %s", e)
        return (
            f"✗ Error during processing: {e}",
            *render_dashboard(), gr.update(visible=False), "", "", "", "", "", gr.update(visible=False), "",
        )

    stat_html, activity_html = render_dashboard()
    return (
        status, stat_html, activity_html,
        gr.update(visible=True), preview_html,
        note.get("chief_complaint", ""), note.get("duration", ""),
        note.get("severity", ""), note.get("history", ""), transcript,
        gr.update(visible=True), confirm_html,
    )


def load_review(choice_label: str | None) -> tuple:
    if not choice_label:
        return (
            "", "", "", "", "", "", "", "",
            gr.update(value="Select a visit to begin review."),
            None,
        )

    visit_id = extract_visit_id(choice_label)
    if visit_id is None:
        return (
            "Invalid selection.", "", "", "", "", "", "", "",
            gr.update(value="Could not parse visit ID from selection."),
            None,
        )

    try:
        visit = db.get_visit_by_id(visit_id)
    except Exception as e:
        return (
            f"Database error: {e}", "", "", "", "", "", "", "",
            gr.update(value=f"Error loading visit: {e}"),
            None,
        )

    if not visit:
        return (
            "Visit not found.", "", "", "", "", "", "", "",
            gr.update(value="Visit not found in database."),
            None,
        )

    header = f"Patient {visit['patient_id']} · {visit['language']} · Visit #{visit['visit_id']} · {visit['created_at']}"
    return (
        header,
        visit["chief_complaint"] or "",
        visit["duration"] or "",
        visit["severity"] or "",
        visit["history"] or "",
        visit["possible_recommendations"] or "",
        format_keywords(visit["extracted_keywords"]),
        visit["raw_transcript"] or "",
        gr.update(value=""),
        visit_id,
    )


def confirm_and_finalize(
    visit_id: int | None,
    chief_complaint: str,
    duration: str,
    severity: str,
    history: str,
    recommendations: str,
) -> tuple[str, str, gr.update, gr.update, int | None]:
    if visit_id is None:
        return "✗ No visit selected.", *render_queue(), gr.update(visible=False), None

    try:
        db.update_visit_and_mark_reviewed(
            visit_id, chief_complaint, duration, severity, history, recommendations
        )
    except Exception as e:
        traceback.print_exc()
        return f"✗ Database error: {e}", *render_queue(), gr.update(visible=False), visit_id

    _cache.invalidate("dashboard")
    _cache.invalidate("queue")

    rows_html, choices = render_queue()
    msg = f"✓ Visit #{visit_id} finalized and marked as reviewed."
    return msg, rows_html, gr.update(choices=choices, value=None), gr.update(visible=False), None


def export_csv(visit_id: int | None) -> str | None:
    if visit_id is None:
        return None
    tmp_path = os.path.join(tempfile.gettempdir(), f"visit_{visit_id}_{int(time.time())}.csv")
    try:
        db.export_single_visit_to_csv(visit_id, tmp_path)
        return tmp_path
    except Exception as e:
        traceback.print_exc()
        return None


# ── navigation helpers ──────────────────────────────────────────────────────
PAGE_NAMES = ["dashboard", "role", "intake", "queue", "review"]


def goto(page_name: str) -> list[gr.update]:
    return [gr.update(visible=(name == page_name)) for name in PAGE_NAMES]


def refresh_dashboard() -> tuple[str, str]:
    return render_dashboard()


def refresh_queue() -> tuple[str, gr.update]:
    rows_html, choices = render_queue()
    return rows_html, gr.update(choices=choices, value=None)


# ── app layout ──────────────────────────────────────────────────────────────
db.init_db()

with gr.Blocks(css=CUSTOM_CSS, title="MediVoice — Multi-Lingual Patient Intake Assistant") as demo:
    visit_id_state = gr.State(None)
    detection_state = gr.State(None)

    with gr.Row():
        # ── sidebar ─────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="sidebar", min_width=240):
            gr.HTML(_sidebar_logo())
            nav_dashboard = gr.Button("🏠  Dashboard", elem_classes=["nav-btn", "active"])
            nav_patients = gr.Button("🧑‍⚕️  Patients", elem_classes=["nav-btn"])
            nav_new_intake = gr.Button("🎙️  New Intake", elem_classes=["nav-btn"])
            gr.HTML(_system_status())

        # ── main content ──────────────────────────────────────────────────
        with gr.Column(scale=4):

            # ---- Dashboard ----
            with gr.Column(visible=True) as dashboard_page:
                gr.HTML(
                    f'<div style="color:{COLOR_TEXT_MUTED};">Good day 👋</div>'
                    f'<div style="font-size:28px; font-weight:700; color:{COLOR_TEXT_DARK};">Patient Intake Dashboard</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Manage patient intake and clinical information efficiently.</div>'
                )
                dash_stats_html = gr.HTML()
                start_intake_btn = gr.Button("Start Intake  →", elem_id="primary-btn")
                gr.HTML(f'<div style="font-size:18px; font-weight:700; color:{COLOR_TEXT_DARK}; margin:20px 0 8px;">Recent activity</div>')
                dash_activity_html = gr.HTML()

            # ---- Role picker ----
            with gr.Column(visible=False) as role_page:
                back_from_role = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="text-align:center; margin-top:20px;">'
                    f'<div style="font-size:22px; font-weight:700; color:{COLOR_TEXT_DARK};">Who\'s using the system?</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:20px;">Select a role to continue</div></div>'
                )
                with gr.Row():
                    role_nurse_btn = gr.Button("🧑‍⚕️  Nurse\nRecord patient intake", elem_classes=["card"])
                    role_doctor_btn = gr.Button("🩺  Doctor\nReview pending visits", elem_classes=["card"])

            # ---- Nurse intake ----
            with gr.Column(visible=False) as intake_page:
                back_from_intake_top = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">New Patient Intake</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Record or upload patient audio to begin.</div>'
                )
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Details</b>')
                    with gr.Row():
                        patient_id_input = gr.Textbox(label="Patient ID", placeholder="e.g. P001", max_lines=1)
                        language_input = gr.Dropdown(
                            LANGUAGES, value=LANGUAGES[0],
                            label="Language (auto-detected — confirm or override)",
                        )
                with gr.Group(elem_classes=["card"]):
                    gr.HTML(f'<b style="color:{COLOR_TEXT_DARK};">Patient Audio</b>')
                    audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or upload")
                    lang_detect_status = gr.Markdown("")
                process_btn = gr.Button("🔄  Transcribe & Structure Note", elem_id="primary-btn")
                intake_status = gr.Markdown("")

                # ── PROCESSING INDICATOR ──
                processing_indicator = gr.Markdown(visible=False)

                # ── NURSE PREVIEW: English translation + raw transcript ──
                with gr.Group(visible=False) as nurse_preview_group:
                    nurse_preview_html = gr.HTML()
                    with gr.Row():
                        nurse_chief = gr.Textbox(label="Chief Complaint", interactive=False)
                        nurse_duration = gr.Textbox(label="Duration", interactive=False)
                    with gr.Row():
                        nurse_severity = gr.Textbox(label="Severity", interactive=False)
                        nurse_history = gr.Textbox(label="History", interactive=False)
                    nurse_transcript = gr.Textbox(label="Original Transcript", interactive=False, lines=3)

                # ── CONFIRMATION BANNER ──
                with gr.Group(visible=False) as confirm_group:
                    confirm_banner_html = gr.HTML()
                    with gr.Row():
                        confirm_back_dashboard = gr.Button("🏠  Return to Dashboard", elem_id="primary-btn")
                        confirm_new_intake = gr.Button("🎙️  Start Another Intake")

                # ── BACK TO DASHBOARD (bottom of page too) ──
                back_from_intake_bottom = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])

            # ---- Doctor queue ----
            with gr.Column(visible=False) as queue_page:
                back_from_queue = gr.Button("← Back to Dashboard", elem_classes=["nav-btn"])
                gr.HTML(
                    f'<div style="font-size:24px; font-weight:700; color:{COLOR_TEXT_DARK}; margin-top:8px;">Patients</div>'
                    f'<div style="color:{COLOR_TEXT_MUTED}; margin-bottom:16px;">Visits awaiting doctor review, oldest first.</div>'
                )
                queue_html = gr.HTML()
                queue_select = gr.Dropdown(label="Select a visit to review", choices=[])
                review_btn = gr.Button("Review →", elem_id="primary-btn")

            # ---- Doctor review ----
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
                finalize_trigger = gr.Markdown(visible=False)

    # ═══════════════════════════════════════════════════════════════════════
    # NAVIGATION WIRING
    # ═══════════════════════════════════════════════════════════════════════
    all_pages = [dashboard_page, role_page, intake_page, queue_page, review_page]

    nav_dashboard.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    nav_new_intake.click(lambda: goto("role"), outputs=all_pages)
    start_intake_btn.click(lambda: goto("role"), outputs=all_pages)
    nav_patients.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    role_nurse_btn.click(lambda: goto("intake"), outputs=all_pages)
    role_doctor_btn.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])
    back_from_role.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_intake_top.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_intake_bottom.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_queue.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    back_from_review.click(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])

    # ═══════════════════════════════════════════════════════════════════════
    # NURSE INTAKE WIRING — with loading indicator & confirmation
    # ═══════════════════════════════════════════════════════════════════════
    audio_input.change(
        run_language_detection,
        inputs=[audio_input],
        outputs=[language_input, lang_detect_status, detection_state],
    )

    # Step 1: Show "Processing..." and disable button
    def _show_processing():
        return (
            gr.update(value='<div class="processing-spinner">🔄 Processing audio... This may take 30–60 seconds. Please do not close this tab.</div>', visible=True),
            gr.update(interactive=False),  # disable process button
        )

    process_btn.click(
        _show_processing,
        outputs=[processing_indicator, process_btn],
    ).then(
        process_intake,
        inputs=[patient_id_input, language_input, audio_input, detection_state],
        outputs=[
            intake_status, dash_stats_html, dash_activity_html,
            nurse_preview_group, nurse_preview_html,
            nurse_chief, nurse_duration, nurse_severity, nurse_history, nurse_transcript,
            confirm_group, confirm_banner_html,
        ],
        show_progress="full",
    ).then(
        # Re-enable button and hide processing indicator after completion
        lambda: (gr.update(visible=False), gr.update(interactive=True)),
        outputs=[processing_indicator, process_btn],
    )

    # Confirmation banner buttons
    confirm_back_dashboard.click(lambda: goto("dashboard"), outputs=all_pages).then(refresh_dashboard, outputs=[dash_stats_html, dash_activity_html])
    confirm_new_intake.click(
        lambda: (
            goto("intake"),
            gr.update(visible=False),  # hide preview
            gr.update(visible=False),  # hide confirm
            gr.update(value=""),       # clear status
            gr.update(value=None),     # clear audio
            gr.update(value=""),       # clear patient id
        ),
        outputs=[
            all_pages[0], all_pages[1], all_pages[2], all_pages[3], all_pages[4],
            nurse_preview_group, confirm_group, intake_status, audio_input, patient_id_input,
        ],
    )

    # ═══════════════════════════════════════════════════════════════════════
    # DOCTOR QUEUE / REVIEW WIRING
    # ═══════════════════════════════════════════════════════════════════════
    review_btn.click(lambda: goto("review"), outputs=all_pages).then(
        load_review,
        inputs=[queue_select],
        outputs=[
            review_header, chief_complaint_box, duration_box, severity_box,
            history_box, recommendations_box, keywords_display, transcript_display,
            review_status, visit_id_state,
        ],
    )

    finalize_btn.click(
        confirm_and_finalize,
        inputs=[visit_id_state, chief_complaint_box, duration_box, severity_box, history_box, recommendations_box],
        outputs=[review_status, queue_html, queue_select, finalize_trigger, visit_id_state],
        show_progress="minimal",
    )

    finalize_trigger.change(lambda: goto("queue"), outputs=all_pages).then(refresh_queue, outputs=[queue_html, queue_select])
    export_btn.click(export_csv, inputs=[visit_id_state], outputs=[export_file])


if __name__ == "__main__":
    demo.launch(share=True)
