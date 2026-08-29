"""
Clinical note database (CSV-backed).
Each patient has a unique ID; all intake visits are linked to that patient
for traceable clinical history.

Storage: CSV files instead of SQLite.
- patients.csv         -> one row per patient
- clinical_visits.csv  -> one row per visit, linked via patient_id
"""
import csv
import json
import os
from pathlib import Path
from datetime import datetime, timezone


# Persist to Google Drive if it's mounted (survives across Colab sessions).
# Falls back to local storage if Drive isn't mounted (data lost when the
# session ends — fine for quick testing, not for real demos).
_DRIVE_DIR = Path("/content/drive/MyDrive/NCAIR-DSA/data")
if os.path.exists("/content/drive/MyDrive"):
    DATA_DIR = _DRIVE_DIR
else:
    DATA_DIR = Path.cwd() / "data"

PATIENTS_CSV = DATA_DIR / "patients.csv"
VISITS_CSV = DATA_DIR / "clinical_visits.csv"

PATIENTS_FIELDS = ["patient_id", "created_at"]
VISITS_FIELDS = [
    "visit_id",
    "patient_id",
    "chief_complaint",
    "duration",
    "severity",
    "history",
    "possible_recommendations",
    "language",
    "extracted_keywords",
    "raw_transcript",
    "created_at",
]

print(f"Data directory: {DATA_DIR}")


def _now():
    return datetime.now(timezone.utc).isoformat()


def init_db():
    """Create the CSV files (with headers) if they don't already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PATIENTS_CSV.exists():
        with open(PATIENTS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=PATIENTS_FIELDS).writeheader()

    if not VISITS_CSV.exists():
        with open(VISITS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=VISITS_FIELDS).writeheader()

    print(f"CSV storage ready at: {DATA_DIR}")


def _patient_exists(patient_id):
    if not PATIENTS_CSV.exists():
        return False
    with open(PATIENTS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(row["patient_id"] == patient_id for row in reader)


def _ensure_patient(patient_id):
    """Insert the patient row if it doesn't already exist (mirrors INSERT OR IGNORE)."""
    if not _patient_exists(patient_id):
        with open(PATIENTS_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=PATIENTS_FIELDS)
            writer.writerow({"patient_id": patient_id, "created_at": _now()})


def _next_visit_id():
    """Mimics AUTOINCREMENT by scanning the existing visit_id column."""
    if not VISITS_CSV.exists():
        return 1
    max_id = 0
    with open(VISITS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                max_id = max(max_id, int(row["visit_id"]))
            except (ValueError, KeyError):
                continue
    return max_id + 1


def _visit_saved_on_disk(visit_id, patient_id):
    """Re-reads the CSV from disk to confirm the row is really there.
    This is the actual proof of persistence — not just 'the write call ran'."""
    if not VISITS_CSV.exists():
        return False
    with open(VISITS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return any(
            row.get("visit_id") == str(visit_id) and row.get("patient_id") == patient_id
            for row in reader
        )


def save_patient_record(patient_id, chief_complaint, duration, severity,
                         history, possible_recommendations, language, keywords, transcript):
    """Save a clinical visit to the patient's history (CSV-backed).

    Returns a confirmation dict the calling application can use to show a
    'saved' message/toast/log line — it only reports success after the row
    has been re-read back off disk, so 'saved' always matches what's
    actually in clinical_visits.csv.
    """
    init_db()  # safe no-op if files already exist; guarantees they're there

    _ensure_patient(patient_id)

    visit_id = _next_visit_id()
    visit_row = {
        "visit_id": visit_id,
        "patient_id": patient_id,
        "chief_complaint": chief_complaint,
        "duration": duration,
        "severity": severity,
        "history": history,
        "possible_recommendations": possible_recommendations,
        "language": language,
        "extracted_keywords": json.dumps(keywords),
        "raw_transcript": transcript,
        "created_at": _now(),
    }

    with open(VISITS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VISITS_FIELDS)
        writer.writerow(visit_row)
        f.flush()
        os.fsync(f.fileno())  # force the write to actually hit disk before we confirm it

    confirmed = _visit_saved_on_disk(visit_id, patient_id)

    result = {
        "saved": confirmed,
        "visit_id": visit_id,
        "patient_id": patient_id,
        "csv_path": str(VISITS_CSV),
        "message": (
            f"Saved visit #{visit_id} for patient {patient_id} to {VISITS_CSV.name}"
            if confirmed
            else f"WARNING: visit #{visit_id} for patient {patient_id} was NOT confirmed in {VISITS_CSV.name}"
        ),
    }

    # This print is what shows the "saved to database" confirmation in the
    # application/notebook — it is tied to the actual on-disk check above,
    # not just to the write call completing.
    print(result["message"])

    return result


def get_patient_history(patient_id):
    """Retrieve full clinical history for a patient, newest visit first."""
    if not VISITS_CSV.exists():
        return []

    with open(VISITS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if row["patient_id"] == patient_id]

    # Sort newest first by created_at (ISO format sorts lexicographically)
    rows.sort(key=lambda r: r["created_at"], reverse=True)

    # Return tuples in the same column order as the original SQLite query
    return [
        (
            row["visit_id"],
            row["chief_complaint"],
            row["duration"],
            row["severity"],
            row["history"],
            row["possible_recommendations"],
            row["language"],
            row["created_at"],
        )
        for row in rows
    ]


init_db()

result = save_patient_record(
    patient_id="P0001",
    chief_complaint="Persistent cough",
    duration="5 days",
    severity="Moderate",
    history="No prior respiratory conditions reported.",
    possible_recommendations="Rest, fluids, follow up if symptoms worsen.",
    language="Igbo",
    keywords=["cough", "5 days", "moderate"],
    transcript="[raw ASR transcript would go here]",
)

result  # inspect the confirmation dict returned to the application


# View the patient's full history back out of the CSV
get_patient_history("P0001")


# Optional: peek at the raw CSV file on disk to see it directly
import pandas as pd
pd.read_csv(VISITS_CSV)


