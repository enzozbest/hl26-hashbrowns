import sqlite3
from pathlib import Path

from data.housing_test_data.parse import parse_data

DB_PATH = Path(__file__).resolve().parent / "housing_test.db"


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS housing_delivery_test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ons_code TEXT,
            council_name TEXT,
            council_id INTEGER,
            homes_required_y1 REAL,
            homes_required_y2 REAL,
            homes_required_y3 REAL,
            total_homes_required REAL,
            homes_delivered_y1 REAL,
            homes_delivered_y2 REAL,
            homes_delivered_y3 REAL,
            total_homes_delivered REAL,
            hdt_measurement REAL,
            hdt_consequence TEXT
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO housing_delivery_test (
            ons_code, council_name, council_id,
            homes_required_y1, homes_required_y2, homes_required_y3,
            total_homes_required,
            homes_delivered_y1, homes_delivered_y2, homes_delivered_y3,
            total_homes_delivered,
            hdt_measurement, hdt_consequence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    init_db(db_path)
    data = parse_data()
    insert_records(data, db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
