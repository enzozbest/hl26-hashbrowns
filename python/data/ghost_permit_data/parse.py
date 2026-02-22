from datetime import datetime
from pathlib import Path

from data.councils import resolve

DATA_DIR = Path(__file__).resolve().parent


def looks_like_date(value):
    """Return True if value looks like a date."""
    try:
        datetime.fromisoformat(value.strip())
        return True
    except Exception:
        return False


def parse_data():
    records = []
    unmatched: set[str] = set()

    files = sorted(DATA_DIR.glob("20*.csv"))

    for file in files:
        with open(str(file), "r", encoding="utf-8", errors="ignore") as f:
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
                    borough_raw = None

                council = resolve(borough_raw) if borough_raw else None
                if council is None:
                    if borough_raw:
                        unmatched.add(borough_raw)
                    continue

                records.append((lpa_number, council.name, council.id))

    if unmatched:
        print(f"  [ghost_permit_data] skipped {len(unmatched)} unmatched borough(s): "
              f"{sorted(unmatched)}")

    return records
