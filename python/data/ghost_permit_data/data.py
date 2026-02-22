import sqlite3
import pandas as pd

from data.ghost_permit_data.create_db import get_db_path

def get_all_data():
    conn = sqlite3.connect(get_db_path())
    df = pd.read_sql_query(
        "SELECT lpa_number, council_name FROM ghost_permit_data",
        conn
    )
    conn.close()
    return df.reset_index(drop=True)


def match_data(df: pd.DataFrame):
    ghost_permits = get_all_data()
    merged = df.merge(
        ghost_permits,
        left_on=["LPA Number", "Borough"],
        right_on=["lpa_number", "council_name"],
        how="left",
        indicator=True
    )
