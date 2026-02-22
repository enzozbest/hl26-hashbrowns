"""Analysis results database.

Separate from planning.db — tracks each analysis run:
  analysis_id  TEXT  (UUID, primary key)
  file_path    TEXT  (directory where PDFs for this run are stored)
  created_at   TEXT  (ISO-8601 UTC timestamp)
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "analysis.db"


def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id TEXT PRIMARY KEY,
            file_path   TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def get_analysis(analysis_id: str, db_path: Path = DB_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT analysis_id, file_path, created_at FROM analyses WHERE analysis_id = ?",
        (analysis_id,),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"analysis_id": row[0], "file_path": row[1], "created_at": row[2]}


def save_analysis(analysis_id: str, file_path: str, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO analyses (analysis_id, file_path, created_at) VALUES (?, ?, ?)",
        (analysis_id, file_path, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()


# Initialise on import so the table always exists before the app handles requests.
init_db()
