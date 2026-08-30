"""
Patient schema + clinical history tracking.
Each patient has a unique ID; all intake visits are linked to that patient
for traceable clinical history.
"""
import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

# Persist to Google Drive if it's mounted (survives across Colab sessions).
# Falls back to local storage if Drive isn't mounted (data lost when the
# Colab session ends — fine for quick testing, not for real demos).
_DRIVE_PATH = Path("/content/drive/MyDrive/NCAIR-DSA/data/notes.db")

if os.path.exists("/content/drive/MyDrive"):
    DB_PATH = _DRIVE_PATH
else:
    DB_PATH = Path(__file__).parent.parent / "data" / "notes.db"


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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database ready at: {DB_PATH}")

def save_patient_record(patient_id, chief_complaint, duration, severity, 
                       history, possible_recommendations, language, keywords, transcript):
    """Save a clinical visit to the patient's history."""
    conn = sqlite3.connect(DB_PATH)
    
    # Ensure patient exists
    conn.execute("INSERT OR IGNORE INTO patients (patient_id) VALUES (?)", (patient_id,))
    
    # Save visit
    conn.execute(
        """INSERT INTO clinical_visits 
           (patient_id, chief_complaint, duration, severity, history, 
            possible_recommendations, language, extracted_keywords, raw_transcript)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (patient_id, chief_complaint, duration, severity, history,
         possible_recommendations, language, json.dumps(keywords), transcript)
    )
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    """Retrieve full clinical history for a patient."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """SELECT visit_id, chief_complaint, duration, severity, history, 
                  possible_recommendations, language, created_at
           FROM clinical_visits
           WHERE patient_id = ?
           ORDER BY created_at DESC""",
        (patient_id,)
    )
    visits = cursor.fetchall()
    conn.close()
    return visits


def export_all_records_to_csv(output_path="patient_records_export.csv"):
    """
    Export the entire clinical_visits table to CSV — every patient, every
    visit, all fields. Meant for handoff to the hospital's own record
    system, or for offline analysis/backup.
    """
    import csv

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("""
        SELECT visit_id, patient_id, chief_complaint, duration, severity,
               history, possible_recommendations, language,
               extracted_keywords, raw_transcript, created_at
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
