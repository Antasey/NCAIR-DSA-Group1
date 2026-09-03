# templates/clinical_note.py — README

## What This Is

The database layer for NCAIR-DSA. SQLite, two tables, linked by patient ID for traceable clinical history across visits. Also handles CSV export of the full dataset.

---

## Schema

**`patients`** — one row per unique patient

| Column | Type | Notes |
|---|---|---|
| `patient_id` | TEXT (primary key) | |
| `created_at` | TIMESTAMP | |

**`clinical_visits`** — one row per visit, linked to a patient

| Column | Type | Notes |
|---|---|---|
| `visit_id` | INTEGER (primary key, autoincrement) | |
| `patient_id` | TEXT | foreign key → `patients` |
| `chief_complaint` | TEXT | from the structured note |
| `duration` | TEXT | from the structured note |
| `severity` | TEXT | from the structured note |
| `history` | TEXT | from the structured note |
| `possible_recommendations` | TEXT | AI-suggested, doctor-editable; hardcoded disclaimer stripped before storage (see `app.py` README) |
| `language` | TEXT | Hausa / Igbo / Yoruba |
| `extracted_keywords` | TEXT | JSON-serialized keyword dict |
| `raw_transcript` | TEXT | original untranslated transcript, kept for reference |
| `created_at` | TIMESTAMP | |

---

## Where the Database File Lives

```python
_DRIVE_PATH = Path("/content/drive/MyDrive/NCAIR-DSA/data/notes.db")

if os.path.exists("/content/drive/MyDrive"):
    DB_PATH = _DRIVE_PATH
else:
    DB_PATH = Path(__file__).parent.parent / "data" / "notes.db"
```

If Google Drive is mounted (checked at import time), the database persists there — surviving across Colab sessions. If not mounted, it falls back to local session storage, which is wiped when the Colab runtime ends. Drive must be mounted from a live notebook cell before `app.py` runs — see the main `app.py` README for why this can't happen inside `app.py` itself.

---

## Migrations — Why They Exist

**The problem this solves:** `CREATE TABLE IF NOT EXISTS` does nothing if the table already exists — it does **not** retroactively add new columns. Early in this project, the schema changed (adding `possible_recommendations`) after some team members already had a `notes.db` file from before that change. The result: `Error saving record: table clinical_visits has no column named possible_recommendations` — a real bug that happened, not a hypothetical.

**The fix:** `init_db()` now calls `_migrate_schema()` every time it runs. This checks the actual columns present in the live database (`PRAGMA table_info`) against a `REQUIRED_COLUMNS` dict, and runs `ALTER TABLE ... ADD COLUMN` for anything missing:

```python
REQUIRED_COLUMNS = {
    "possible_recommendations": "TEXT",
    # Add future new columns here as: "column_name": "SQL_TYPE"
}
```

**If the schema changes again** (a new field gets added to the clinical note, for example), add one line to `REQUIRED_COLUMNS` — every existing database, on every team member's Drive, will pick up the new column automatically the next time `app.py` runs `init_db()`. No one needs to manually delete their `notes.db` or run an `ALTER TABLE` by hand again.

**What this does NOT handle:** renaming or removing a column, or changing a column's type. Those are more invasive schema changes that this simple migration approach isn't built for — if that's ever needed, it would require a proper migration (create new table, copy data across, drop old table), not just an `ADD COLUMN`.

---

## Functions

### `init_db()`
Creates both tables if they don't exist, then runs the migration check. Called once at the top of `app.py`, before the Gradio interface is built. Safe to call multiple times — it's idempotent (running it twice does nothing the second time, doesn't duplicate data or error).

### `save_patient_record(patient_id, chief_complaint, duration, severity, history, possible_recommendations, language, keywords, transcript)`
Inserts a new visit. Also inserts the patient into the `patients` table if they don't already exist (`INSERT OR IGNORE`), so a brand-new patient ID doesn't need a separate "register patient" step first.

### `get_patient_history(patient_id)`
Returns all visits for a patient, most recent first. Returns an empty list (not an error) for a patient ID with no visits.

### `export_all_records_to_csv(output_path="patient_records_export.csv")`
Dumps the entire `clinical_visits` table — every patient, every visit, every column — to a CSV file. Returns `(output_path, row_count)`. Used by the Export tab in `app.py` (Doctor role only).

---

## Testing

Covered in `tests/test_pipeline.py`:
- Table creation
- Save + retrieve for a single visit
- Multiple visits for the same patient (history accumulates correctly)
- Unknown patient ID returns empty history, not an error
- Visits are ordered most-recent-first

The migration logic specifically was manually verified by simulating an old-schema database (missing `possible_recommendations`), confirming `init_db()` detects and adds the column, and confirming a save succeeds afterward without needing to delete anything.

---

## Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `table clinical_visits has no column named X` | Should no longer happen — the migration in `init_db()` handles this automatically as of this version. If it still occurs, check that `_migrate_schema()` is actually being called (i.e. `init_db()` wasn't skipped or an older version of this file is in use). | Confirm this file matches what's in the repo; re-run `init_db()` |
| `sqlite3.DatabaseError: file is not a database` | The `.db` file is corrupted or empty (not a schema issue — this is a different failure mode, migrations can't fix a broken file) | Delete the file and let `init_db()` recreate it fresh: `!rm -f <DB_PATH>` |
| Data disappears between Colab sessions | Google Drive wasn't mounted before `app.py` ran, so `DB_PATH` fell back to local (session-only) storage | Mount Drive first, in its own notebook cell, before running `app.py` |
