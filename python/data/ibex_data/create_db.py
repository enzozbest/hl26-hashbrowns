"""Create and populate the Ibex API data cache.

Stores planning application data and council statistics fetched from the
Ibex API so that repeated training runs can reuse previously-fetched data
instead of hitting the API every time.

Two tables:
  - ``ibex_applications``  — mirrors ``ApplicationsResponse`` fields
  - ``ibex_council_stats`` — mirrors ``StatsResponse`` fields

Uses ``INSERT OR REPLACE`` on UNIQUE constraints so re-running accumulates
new data and updates existing records.
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "ibex.db"


def get_db_path():
    return DB_PATH


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ibex_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            council_id INTEGER NOT NULL,
            council_name TEXT,
            planning_reference TEXT,
            url TEXT,
            proposal TEXT,
            raw_address TEXT,
            raw_application_type TEXT,
            normalised_application_type TEXT,
            normalised_decision TEXT,
            raw_decision TEXT,
            application_date TEXT,
            decided_date TEXT,
            geometry TEXT,
            project_type TEXT,
            heading TEXT,
            num_new_houses INTEGER,
            num_comments_received INTEGER,
            proposed_unit_mix TEXT,
            proposed_floor_area TEXT,
            document_metadata TEXT,
            appeals TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(council_id, planning_reference)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ibex_council_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            council_id INTEGER NOT NULL,
            approval_rate REAL,
            refusal_rate REAL,
            average_decision_time TEXT,
            number_of_applications TEXT,
            number_of_new_homes_approved INTEGER,
            council_development_activity_level TEXT,
            date_from TEXT,
            date_to TEXT,
            fetched_at TEXT NOT NULL,
            UNIQUE(council_id, date_from, date_to)
        )
    """)

    conn.commit()
    conn.close()


def insert_application_records(records, db_path):
    """Upsert application records into ibex_applications."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT OR REPLACE INTO ibex_applications (
            council_id, council_name, planning_reference, url,
            proposal, raw_address, raw_application_type,
            normalised_application_type, normalised_decision,
            raw_decision, application_date, decided_date,
            geometry, project_type, heading,
            num_new_houses, num_comments_received,
            proposed_unit_mix, proposed_floor_area,
            document_metadata, appeals, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def insert_stats_records(records, db_path):
    """Upsert stats records into ibex_council_stats."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT OR REPLACE INTO ibex_council_stats (
            council_id, approval_rate, refusal_rate,
            average_decision_time, number_of_applications,
            number_of_new_homes_approved,
            council_development_activity_level,
            date_from, date_to, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    """Initialise the Ibex cache tables (no data to pre-populate)."""
    init_db(db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
