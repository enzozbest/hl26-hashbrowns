import sqlite3
import zipfile
from pathlib import Path

from data.flood_data.parse import parse_data, GPKG_PATH

DATA_DIR = Path(__file__).resolve().parent
ZIP_PATH = DATA_DIR / "Flood_Map_for_Planning_Flood_Zones.gpkg.zip"
DB_PATH = DATA_DIR / "flood.db"

DOWNLOAD_INSTRUCTIONS = """
Flood zone data not found. Please download it manually:

  1. Go to: https://www.data.gov.uk/dataset/bed63fc1-dd26-4685-b143-2941088923b3/flood-map-for-planning-flood-zones
  2. Download: Flood_Map_for_Planning_Flood_Zones.gpkg.zip
  3. Place the zip file at:
     {zip_path}

Then re-run this script.
"""


def ensure_gpkg():
    """Unzip the geopackage if needed. Returns True if ready, False if missing."""
    if GPKG_PATH.exists():
        return True

    if ZIP_PATH.exists():
        print(f"  Unzipping {ZIP_PATH.name}...")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        if GPKG_PATH.exists():
            return True

    print(DOWNLOAD_INSTRUCTIONS.format(zip_path=ZIP_PATH))
    return False


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flood_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin TEXT,
            flood_zone TEXT,
            flood_source TEXT,
            shape_length REAL,
            shape_area REAL,
            geometry BLOB
        )
    """)

    conn.commit()
    conn.close()


def insert_records(records, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executemany("""
        INSERT INTO flood_zones (
            origin, flood_zone, flood_source,
            shape_length, shape_area, geometry
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()


def populate_db(db_path):
    if not ensure_gpkg():
        return

    print("  Reading flood zone geopackage (this may take a while)...")
    init_db(db_path)
    data = parse_data()
    print(f"  Inserting {len(data):,} flood zone features...")
    insert_records(data, db_path)


if __name__ == "__main__":
    populate_db(DB_PATH)
