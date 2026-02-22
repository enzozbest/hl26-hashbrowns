import sqlite3
from pathlib import Path

from data.green_belt_data.parse import parse_data

DB_PATH = Path(__file__).resolve().parent / "green_belt.db"


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS green_belt (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity INTEGER,
            council_name TEXT,
            reference TEXT,
            green_belt_core TEXT,
            ons_code TEXT,
            entry_date TEXT,
            start_date TEXT,
            end_date TEXT,
            quality TEXT,
            geometry BLOB
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO green_belt (
            entity, council_name, reference, green_belt_core,
            ons_code, entry_date, start_date, end_date,
            quality, geometry
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    init_db(db_path)
    data = parse_data()
    insert_records(data, db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
