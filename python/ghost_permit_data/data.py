import sqlite3
import pandas as pd

from ghost_permit_data.create_db import get_db_path

def get_all_data():
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query(
        "SELECT lpa_number, borough FROM planning_data",
        conn
    )
    conn.close()
    return df
