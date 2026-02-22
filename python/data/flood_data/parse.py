from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parent
GPKG_PATH = DATA_DIR / "Flood_Map_for_Planning_Flood_Zones.gpkg"


def parse_data():
    gdf = gpd.read_file(GPKG_PATH)

    records = []
    for row in gdf.itertuples(index=False):
        records.append((
            row.origin or None,
            row.flood_zone or None,
            row.flood_source if row.flood_source == row.flood_source else None,  # NaN check
            row.shape_length if row.shape_length == row.shape_length else None,
            row.shape_area if row.shape_area == row.shape_area else None,
            row.geometry.wkb if row.geometry is not None else None,
        ))

    return records
