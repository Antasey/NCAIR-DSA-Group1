"""
Patient schema + clinical history tracking, with a review queue.

Each patient has a unique ID; all intake visits are linked to that patient
for traceable clinical history. Visits carry a status so the app can
distinguish what a nurse has captured but a doctor hasn't reviewed yet
("pending_review") from what's been finalized ("reviewed") — this is what
powers the Doctor's review queue in the desktop app.

Fully offline / local-first: no Google Drive dependency (this was a Colab
convenience from an earlier version of the project; the desktop app stores
its database in a local app-data folder instead, since it runs entirely
on-device).
"""
import sqlite3
import json
import csv
from pathlib import Path
from datetime import datetime

# Local app-data folder — works the same regardless of OS, no cloud
# dependency, since this is a fully offline desktop app.
DB_PATH = Path(__file__).parent.parent / "data" / "notes.db"

STATUS_PENDING = "pending_review"
STATUS_REVIEWED = "reviewed"


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    # Patients table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Clinical visits table (linked to patient)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clinical_visits (
            visit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            chief_complaint TEXT,
            duration TEXT,
            severity TEXT,
            history TEXT,
            possible_recommendations TEXT,
            language TEXT,
            extracted_keywords TEXT,
            raw_transcript TEXT,
            status TEXT DEFAULT 'pending_review',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewed_at TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)

    conn.commit()
    conn.close()

    _migrate_schema()
    print(f"Database ready at: {DB_PATH}")


def _migrate_schema():
    """
    Real migration step — runs every time init_db() is called.

    CREATE TABLE IF NOT EXISTS does NOT retroactively add new columns to a
    table that already exists. Without this, anyone with an older notes.db
    (created before a field was added — e.g. possible_recommendations,
    or now status/reviewed_at) gets silent failures on save instead of the
    schema quietly catching up.

    Add any future new column to REQUIRED_COLUMNS below when the schema
    changes again; existing databases pick it up automatically on the next
    init_db() call, no manual deletion or ALTER TABLE needed by hand.
    """
    REQUIRED_COLUMNS = {
        "possible_recommendations": "TEXT",
        "status": "TEXT DEFAULT 'pending_review'",
        "reviewed_at": "TIMESTAMP",
        # Add future new columns here as: "column_name": "SQL_TYPE"
    }

    conn = sqlite3.connect(DB_PATH)
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(clinical_visits)").fetchall()
    }

    added_any = False
    for column_name, column_type in REQUIRED_COLUMNS.items():
        if column_name not in existing_columns:
            conn.execute(f"ALTER TABLE clinical_visits ADD COLUMN {column_name} {column_type}")
            print(f"Migration: added missing column '{column_name}' to clinical_visits")
            added_any = True

    if added_any:
        conn.commit()
    conn.close()


def save_patient_record(patient_id, chief_complaint, duration, severity,
                         history, possible_recommendations, language, keywords, transcript):
    """
    Save a clinical visit to the patient's history. New visits always start
    as 'pending_review' — this is the Nurse-side save, before any doctor
    has looked at it.
    """
    conn = sqlite3.connect(DB_PATH)

    conn.execute("INSERT OR IGNORE INTO patients (patient_id) VALUES (?)", (patient_id,))

    conn.execute(
        """INSERT INTO clinical_visits
           (patient_id, chief_complaint, duration, severity, history,
            possible_recommendations, language, extracted_keywords, raw_transcript, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, chief_complaint, duration, severity, history,
         possible_recommendations, language, json.dumps(keywords), transcript, STATUS_PENDING)
    )
    conn.commit()
    visit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return visit_id


def update_visit_and_mark_reviewed(visit_id, chief_complaint, duration, severity,
                                    history, possible_recommendations):
    """
    Doctor-side save: applies the doctor's edits to an existing visit and
    marks it reviewed. This is what "Confirm & Finalize" in the Doctor
    review screen calls — distinct from save_patient_record(), which is
    only ever used for the initial Nurse-side capture.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """UPDATE clinical_visits
           SET chief_complaint = ?, duration = ?, severity = ?, history = ?,
               possible_recommendations = ?, status = ?, reviewed_at = CURRENT_TIMESTAMP
           WHERE visit_id = ?""",
        (chief_complaint, duration, severity, history, possible_recommendations,
         STATUS_REVIEWED, visit_id)
    )
    conn.commit()
    conn.close()


def get_patient_history(patient_id):
    """Retrieve full clinical history for a patient, most recent first."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT visit_id, chief_complaint, duration, severity, history,
                  possible_recommendations, language, status, created_at
           FROM clinical_visits
           WHERE patient_id = ?
           ORDER BY created_at DESC""",
        (patient_id,)
    )
    visits = cursor.fetchall()
    conn.close()
    return visits


def get_pending_visits():
    """
    Return all visits still awaiting doctor review, oldest first (so the
    doctor naturally works through the queue in the order patients arrived).
    Powers the Doctor's Patients/review-queue screen.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT visit_id, patient_id, chief_complaint, language, created_at
           FROM clinical_visits
           WHERE status = ?
           ORDER BY created_at ASC""",
        (STATUS_PENDING,)
    )
    visits = cursor.fetchall()
    conn.close()
    return visits


def get_visit_by_id(visit_id):
    """Fetch a single visit's full detail — used when the doctor opens one
    from the review queue."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT visit_id, patient_id, chief_complaint, duration, severity,
                  history, possible_recommendations, language,
                  extracted_keywords, raw_transcript, status, created_at
           FROM clinical_visits
           WHERE visit_id = ?""",
        (visit_id,)
    )
    row = cursor.fetchone()
    conn.close()
    return row


def get_dashboard_stats():
    """Counts for the Dashboard stat cards."""
    conn = sqlite3.connect(DB_PATH)
    total_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_visits = conn.execute("SELECT COUNT(*) FROM clinical_visits").fetchone()[0]
    pending_count = conn.execute(
        "SELECT COUNT(*) FROM clinical_visits WHERE status = ?", (STATUS_PENDING,)
    ).fetchone()[0]
    conn.close()
    return {
        "total_patients": total_patients,
        "total_visits": total_visits,
        "pending_review": pending_count,
    }


# ============================================================================
# Exports — CSV only. A .txt export existed in an earlier version; replaced
# entirely with CSV so both exports open cleanly in Excel/Sheets/Numbers
# instead of a plain text file, and so what's actually stored is visible
# in a format non-technical staff can read without confusion.
# ============================================================================

def export_all_records_to_csv(output_path="patient_records_export.csv"):
    """Export the entire clinical_visits table to CSV — every patient,
    every visit, every column."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT visit_id, patient_id, chief_complaint, duration, severity,
               history, possible_recommendations, language,
               extracted_keywords, raw_transcript, status, created_at, reviewed_at
        FROM clinical_visits
        ORDER BY patient_id, created_at DESC
    """)
    rows = cursor.fetchall()
    column_names = [description[0] for description in cursor.description]
    conn.close()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerows(rows)

    return output_path, len(rows)


def export_single_visit_to_csv(visit_id, output_path=None):
    """
    Export one visit as CSV — replaces the earlier plain-text transcript
    export. One row, all fields as columns, including the original
    untranslated transcript (kept separately so a mistranslation never
    loses the source text).
    """
    row = get_visit_by_id(visit_id)
    if not row:
        raise ValueError(f"No visit found with visit_id={visit_id}")

    column_names = [
        "visit_id", "patient_id", "chief_complaint", "duration", "severity",
        "history", "possible_recommendations", "language",
        "extracted_keywords", "raw_transcript", "status", "created_at"
    ]

    if output_path is None:
        output_path = f"visit_{visit_id}_export.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerow(row)

    return output_path
