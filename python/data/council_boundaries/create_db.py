"""Download and store English local authority district boundary polygons.

Source: martinjc/UK-GeoJSON (WGS84 / CRS84)
Table:  council_boundaries
  ons_code     TEXT  - ONS/LAD code (unique key)
  lad_name     TEXT  - raw LAD13NM name from GeoJSON (or merge key)
  council_name TEXT  - canonical name
  council_id   INT   - internal registry ID
  region       STR   - Region the council is in
  geometry     BLOB  - WKB for spatial analysis (green belt section etc.)
  feature_json TEXT  - pre-serialised GeoJSON feature for zero-cost API serving

LADs listed in MERGE_REGIONS are dissolved into a single polygon named after
the merge key (e.g. the four old Dorset districts → one 'Dorset' polygon).
"""
import json
import sqlite3
import urllib.request
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from data.councils import resolve
from data.regions import COUNCIL_REGION, MERGE_REGIONS

_URL = "https://raw.githubusercontent.com/martinjc/UK-GeoJSON/master/json/administrative/eng/lad.json"
DB_PATH = Path(__file__).resolve().parent.parent / "planning.db"

# Reverse lookup: LAD13NM → merge key (e.g. 'East Dorset' → 'Dorset')
_LAD_TO_MERGE_KEY: dict[str, str] = {
    lad_name: merge_key
    for merge_key, lad_names in MERGE_REGIONS.items()
    for lad_name in lad_names
}


def _download_geojson() -> dict:
    with urllib.request.urlopen(_URL) as response:
        return json.load(response)


def _get_ons_code(db_path, council_name: str) -> str | None:
    """Look up the current ONS code for a council from income_data."""
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT ons_code FROM income_data WHERE council_name = ?", (council_name,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS council_boundaries")
    conn.execute("""
        CREATE TABLE council_boundaries (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ons_code     TEXT UNIQUE,
            lad_name     TEXT,
            council_name TEXT,
            council_id   INTEGER,
            region       TEXT,
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

    # Accumulate geometries for merge groups keyed by merge key name
    merge_geoms: dict[str, list] = {key: [] for key in MERGE_REGIONS}

    for feature in features:
        props = feature["properties"]
        ons_code = props["LAD13CD"]
        raw_name = props["LAD13NM"]

        # If this LAD belongs to a merge group, collect its geometry and skip
        merge_key = _LAD_TO_MERGE_KEY.get(raw_name)
        if merge_key is not None:
            merge_geoms[merge_key].append(shape(feature["geometry"]))
            continue

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

    # Build one merged record per merge group
    for merge_key, geoms in merge_geoms.items():
        if not geoms:
            continue

        council = resolve(merge_key)
        if council is None:
            print(f"  [council_boundaries] could not resolve merge key '{merge_key}', skipping")
            continue

        merged_geom = unary_union(geoms)
        ons_code = _get_ons_code(db_path, council.name)
        region = COUNCIL_REGION.get(merge_key)

        feature_json = json.dumps({
            "ons_code": ons_code,
            "council_name": council.name,
            "council_id": council.id,
            "geometry": mapping(merged_geom),
        })
        records.append((ons_code, merge_key, council.name, council.id, region, merged_geom.wkb, feature_json))

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
