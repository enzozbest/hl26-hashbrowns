"""Download and store English local authority district boundary polygons.

Source: martinjc/UK-GeoJSON (WGS84 / CRS84)
Table:  council_boundaries
  ons_code     TEXT  - ONS/LAD code (unique key)
  council_name TEXT  - canonical name
  council_id   INT   - internal registry ID
  region       STR   - Region the council is in
  geometry     BLOB  - WKB for spatial analysis (green belt section etc.)
  feature_json TEXT  - pre-serialised GeoJSON feature for zero-cost API serving
"""
import json
import sqlite3
import urllib.request
from pathlib import Path

from shapely.geometry import shape

from data.councils import resolve
from data.regions import COUNCIL_REGION

_URL = "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/eng/lad.json"
DB_PATH = Path(__file__).resolve().parent.parent / "planning.db"


def _download_geojson() -> dict:
    with urllib.request.urlopen(_URL) as response:
        return json.load(response)


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS council_boundaries")
    conn.execute("""
        CREATE TABLE council_boundaries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ons_code     TEXT UNIQUE,
            lad_name TEXT,
            council_name TEXT,
            council_id   INTEGER,
            region TEXT,
            geometry     BLOB,
            feature_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def populate_db(db_path):
    init_db(db_path)

    print("  Downloading council boundary GeoJSON...")
    geojson = _download_geojson()
    features = geojson["features"]

    records = []
    unmatched: set[str] = set()
    no_region: set[str] = set()

    for feature in features:
        props = feature["properties"]
        ons_code = props["LAD13CD"]
        raw_name = props["LAD13NM"]

        council = resolve(ons_code) or resolve(raw_name)
        if council is None:
            unmatched.add(raw_name)
            continue

        region = COUNCIL_REGION.get(raw_name)
        if region is None:
            no_region.add(raw_name)

        geom = shape(feature["geometry"])
        feature_json = json.dumps({
            "ons_code": ons_code,
            "council_name": council.name,
            "council_id": council.id,
            "geometry": feature["geometry"],
        })

        records.append((ons_code, raw_name, council.name, council.id, region, geom.wkb, feature_json))

    if unmatched:
        print(f"  [council_boundaries] skipped {len(unmatched)} unmatched: {sorted(unmatched)}")
    if no_region:
        print(f"  [council_boundaries] no region for {len(no_region)}: {sorted(no_region)}")

    conn = sqlite3.connect(db_path)
    conn.executemany("""
        INSERT OR REPLACE INTO council_boundaries
            (ons_code, lad_name, council_name, council_id, region, geometry, feature_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, records)
    conn.commit()
    conn.close()

    print(f"  [council_boundaries] inserted {len(records)} boundaries")


if __name__ == "__main__":
    populate_db(DB_PATH)
