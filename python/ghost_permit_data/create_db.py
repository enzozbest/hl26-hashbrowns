import sqlite3
from pathlib import Path

from ghost_permit_data.parse import parse_data

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "planning_applications.db"

def get_db_path():
    return DB_PATH


def init_db():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS planning_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lpa_number TEXT,
            borough TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records):
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO planning_data (lpa_number, borough)
        VALUES (?, ?)
    """, records)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    data = parse_data()
    insert_records(data)