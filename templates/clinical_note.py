"""
Patient schema + clinical history tracking, with a review queue.

Each patient has a unique ID; all intake visits are linked to that patient
for traceable clinical history. Visits carry a status so the app can
distinguish pending from reviewed.

Fully offline / local-first: no cloud dependency.
"""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Local app-data folder — works the same regardless of OS
DB_PATH: Final = Path(__file__).parent.parent / "data" / "notes.db"

STATUS_PENDING: Final = "pending_review"
STATUS_REVIEWED: Final = "reviewed"


# ============================================================================
# CONNECTION MANAGEMENT
# ============================================================================

@contextmanager
def _get_connection():
    """Yield a SQLite connection with row factory and foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ============================================================================
# SCHEMA & MIGRATIONS
# ============================================================================

def init_db() -> None:
    """Create tables if they don't exist, then run migration check."""
    with _get_connection() as conn:
        # Patients table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clinical visits table
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

        # Indexes for common query patterns
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_patient ON clinical_visits(patient_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_status ON clinical_visits(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visits_created ON clinical_visits(created_at)")

        conn.commit()

    _migrate_schema()
    logger.info("Database ready at: %s", DB_PATH)


def _migrate_schema() -> None:
    """Add any missing columns to existing tables."""
    REQUIRED_COLUMNS = {
        "possible_recommendations": "TEXT",
        "status": "TEXT DEFAULT 'pending_review'",
        "reviewed_at": "TIMESTAMP",
    }

    with _get_connection() as conn:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(clinical_visits)").fetchall()
        }

        for col_name, col_type in REQUIRED_COLUMNS.items():
            if col_name not in existing:
                conn.execute(f"ALTER TABLE clinical_visits ADD COLUMN {col_name} {col_type}")
                logger.info("Migration: added column '%s' to clinical_visits", col_name)

        conn.commit()


# ============================================================================
# CRUD OPERATIONS
# ============================================================================

def save_patient_record(
    patient_id: str,
    chief_complaint: str,
    duration: str,
    severity: str,
    history: str,
    possible_recommendations: str,
    language: str,
    keywords: dict,
    transcript: str,
) -> int:
    """
    Save a clinical visit. New visits always start as 'pending_review'.
    Uses a single transaction for patient insert + visit insert.
    """
    with _get_connection() as conn:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO patients (patient_id) VALUES (?)",
                (patient_id,),
            )
            conn.execute(
                """INSERT INTO clinical_visits
                   (patient_id, chief_complaint, duration, severity, history,
                    possible_recommendations, language, extracted_keywords,
                    raw_transcript, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    patient_id, chief_complaint, duration, severity, history,
                    possible_recommendations, language, json.dumps(keywords),
                    transcript, STATUS_PENDING,
                ),
            )
            conn.commit()
            visit_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return visit_id
        except Exception:
            conn.rollback()
            raise


def update_visit_and_mark_reviewed(
    visit_id: int,
    chief_complaint: str,
    duration: str,
    severity: str,
    history: str,
    possible_recommendations: str,
) -> None:
    """Doctor-side save: applies edits and marks visit as reviewed."""
    with _get_connection() as conn:
        conn.execute(
            """UPDATE clinical_visits
               SET chief_complaint = ?, duration = ?, severity = ?, history = ?,
                   possible_recommendations = ?, status = ?, reviewed_at = ?
               WHERE visit_id = ?""",
            (
                chief_complaint, duration, severity, history,
                possible_recommendations, STATUS_REVIEWED,
                datetime.now().isoformat(), visit_id,
            ),
        )
        conn.commit()


def get_patient_history(patient_id: str) -> list[sqlite3.Row]:
    """Retrieve full clinical history for a patient, most recent first."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """SELECT visit_id, chief_complaint, duration, severity, history,
                      possible_recommendations, language, status, created_at
               FROM clinical_visits
               WHERE patient_id = ?
               ORDER BY created_at DESC""",
            (patient_id,),
        )
        return cursor.fetchall()


def get_pending_visits() -> list[sqlite3.Row]:
    """Return all visits awaiting doctor review, oldest first."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """SELECT visit_id, patient_id, chief_complaint, language, created_at
               FROM clinical_visits
               WHERE status = ?
               ORDER BY created_at ASC""",
            (STATUS_PENDING,),
        )
        return cursor.fetchall()


def get_visit_by_id(visit_id: int) -> sqlite3.Row | None:
    """Fetch a single visit's full detail."""
    with _get_connection() as conn:
        cursor = conn.execute(
            """SELECT visit_id, patient_id, chief_complaint, duration, severity,
                      history, possible_recommendations, language,
                      extracted_keywords, raw_transcript, status, created_at
               FROM clinical_visits
               WHERE visit_id = ?""",
            (visit_id,),
        )
        return cursor.fetchone()


def get_dashboard_stats() -> dict[str, int]:
    """Counts for the Dashboard stat cards — single query."""
    with _get_connection() as conn:
        row = conn.execute(
            """SELECT
                (SELECT COUNT(*) FROM patients) AS total_patients,
                (SELECT COUNT(*) FROM clinical_visits) AS total_visits,
                (SELECT COUNT(*) FROM clinical_visits WHERE status = ?) AS pending_review
            """,
            (STATUS_PENDING,),
        ).fetchone()
        return {
            "total_patients": row["total_patients"],
            "total_visits": row["total_visits"],
            "pending_review": row["pending_review"],
        }


# ============================================================================
# EXPORTS
# ============================================================================

def export_all_records_to_csv(output_path: str = "patient_records_export.csv") -> tuple[str, int]:
    """Export the entire clinical_visits table to CSV."""
    with _get_connection() as conn:
        cursor = conn.execute("""
            SELECT visit_id, patient_id, chief_complaint, duration, severity,
                   history, possible_recommendations, language,
                   extracted_keywords, raw_transcript, status, created_at, reviewed_at
            FROM clinical_visits
            ORDER BY patient_id, created_at DESC
        """)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([d[0] for d in cursor.description])
            writer.writerows(cursor)

        # Re-count rows without loading everything into memory
        count = conn.execute("SELECT COUNT(*) FROM clinical_visits").fetchone()[0]

    return output_path, count


def export_single_visit_to_csv(visit_id: int, output_path: str | None = None) -> str:
    """Export one visit as CSV."""
    row = get_visit_by_id(visit_id)
    if not row:
        raise ValueError(f"No visit found with visit_id={visit_id}")

    column_names = [
        "visit_id", "patient_id", "chief_complaint", "duration", "severity",
        "history", "possible_recommendations", "language",
        "extracted_keywords", "raw_transcript", "status", "created_at",
    ]

    if output_path is None:
        output_path = f"visit_{visit_id}_export.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        writer.writerow(tuple(row))

    return output_path
