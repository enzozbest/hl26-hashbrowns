import glob
from datetime import datetime

import pandas as pd


def looks_like_date(value):
    """Return True if value looks like a date."""
    try:
        datetime.fromisoformat(value.strip())
        return True
    except Exception:
        return False


def parse_data():
    records = []

    files = sorted(glob.glob("20*.csv"))

    for file in files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            next(f)  # skip header

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Split only first 2 commas
                parts = line.split(",", 2)

                lpa_number = parts[0].strip() if len(parts) > 0 else None
                borough_raw = parts[1].strip() if len(parts) > 1 else None

                if borough_raw and looks_like_date(borough_raw):
                    borough = None
                else:
                    borough = borough_raw

                records.append((lpa_number, borough))

    return records

