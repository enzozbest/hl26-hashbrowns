import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "planning.db"


def query(table: str, sql: str = "SELECT *", query: str = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    if query:
        df = pd.read_sql_query(query, conn)
    else:
        df = pd.read_sql_query(f"{sql} FROM {table}", conn)
    conn.close()
    return df
