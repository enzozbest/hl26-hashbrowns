from pathlib import Path

import pandas as pd

from data.councils import resolve

DATA_DIR = Path(__file__).resolve().parent


def parse_data():
    records = []
    unmatched: set[str] = set()

    for path in sorted(DATA_DIR.glob("*.ods")):
        df = pd.read_excel(path, engine="odf", header=None)

        # Rows 0-5 are title/header rows; data starts at row 6
        data = df.iloc[6:].reset_index(drop=True)

        for _, row in data.iterrows():
            ons_code = row[0]
            area_name = row[1]

            if pd.isna(ons_code) or pd.isna(area_name):
                continue

            council = resolve(str(area_name))
            if council is None:
                unmatched.add(str(area_name))
                continue

            records.append((
                str(ons_code),
                council.name,          # normalised area_name
                council.id,            # council_id
                _float_or_none(row[2]),   # homes_required_y1
                _float_or_none(row[3]),   # homes_required_y2
                _float_or_none(row[4]),   # homes_required_y3
                _float_or_none(row[5]),   # total_homes_required
                _float_or_none(row[6]),   # homes_delivered_y1
                _float_or_none(row[7]),   # homes_delivered_y2
                _float_or_none(row[8]),   # homes_delivered_y3
                _float_or_none(row[9]),   # total_homes_delivered
                _float_or_none(row[10]),  # hdt_measurement
                _str_or_none(row[11]),    # hdt_consequence
            ))

    if unmatched:
        print(f"  [housing_test_data] skipped {len(unmatched)} unmatched area(s): "
              f"{sorted(unmatched)}")

    return records


def _float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value):
    if pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None
