import sqlite3
from pathlib import Path

from data.housing_test_data.parse import parse_data

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "housing_test.db"


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS housing_delivery_test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year TEXT,
            ons_code TEXT,
            area_name TEXT,
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
            year, ons_code, area_name,
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
