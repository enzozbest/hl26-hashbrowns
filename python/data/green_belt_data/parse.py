from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent
GEOJSON_PATH = DATA_DIR / "green-belt.geojson"


def _date_str(value):
    if value is None:
        return None
    try:
        return value.date().isoformat()
    except AttributeError:
        s = str(value).strip()
        return s if s else None


def parse_data():
    gdf = gpd.read_file(GEOJSON_PATH)
    gdf = gdf.rename(columns={
        "end-date": "end_date",
        "entry-date": "entry_date",
        "start-date": "start_date",
        "green-belt-core": "green_belt_core",
        "local-authority-district": "ons_code",
        "name": "council_name",
    })

    records = []
    for row in gdf.itertuples(index=False):
        records.append((
            row.entity,
            row.council_name or None,
            row.reference or None,
            row.green_belt_core or None,
            row.ons_code or None,
            _date_str(row.entry_date),
            _date_str(row.start_date),
            _date_str(row.end_date),
            row.quality or None,
            row.geometry.wkb if row.geometry is not None else None,
        ))

    return records
