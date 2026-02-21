import sqlite3
from pathlib import Path

from data.ghost_permit_data.parse import parse_data

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "planning_applications.db"

def get_db_path():
    return DB_PATH


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ghost_permit_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lpa_number TEXT,
            borough TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO ghost_permit_data (lpa_number, borough)
        VALUES (?, ?)
    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    init_db(db_path)
    data = parse_data()
    insert_records(data, db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
