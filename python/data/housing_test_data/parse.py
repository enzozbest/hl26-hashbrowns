from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def parse_data():
    records = []

    for path in sorted(DATA_DIR.glob("*.ods")):
        year = path.stem  # e.g. "2023"
        df = pd.read_excel(path, engine="odf", header=None)

        # Rows 0-5 are title/header rows; data starts at row 6
        data = df.iloc[6:].reset_index(drop=True)

        for _, row in data.iterrows():
            ons_code = row[0]
            area_name = row[1]

            if pd.isna(ons_code) or pd.isna(area_name):
                continue

            records.append((
                str(year),
                str(ons_code),
                str(area_name),
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
