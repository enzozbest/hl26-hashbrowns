import sqlite3
from pathlib import Path

from data.income_data.parse import parse_data

DB_PATH = Path(__file__).resolve().parent / "income.db"


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS income_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            year TEXT,
            local_authority_code TEXT,
            local_authority_name TEXT,
            region_name TEXT,
            mean_disposable_income REAL
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO income_data (
            year, local_authority_code, local_authority_name,
            region_name, mean_disposable_income
        ) VALUES (?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    init_db(db_path)
    data = parse_data()
    insert_records(data, db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
