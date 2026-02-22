"""
Top-level script to build the shared planning database.

Creates data/planning.db and populates it by calling each
data source's populate_db function.
"""
from pathlib import Path

from data.council_boundaries.create_db import populate_db as populate_council_boundaries
from data.ghost_permit_data.create_db import populate_db as populate_ghost_permits
from data.green_belt_data.create_db import populate_db as populate_green_belt
from data.housing_test_data.create_db import populate_db as populate_housing_test
from data.ibex_data.create_db import populate_db as populate_ibex
from data.income_data.create_db import populate_db as populate_income

DB_PATH = Path(__file__).resolve().parent / "planning.db"


def build_db():
    print(f"Building database at {DB_PATH}")

    print("  Populating ghost permit data...")
    populate_ghost_permits(DB_PATH)

    print("  Populating housing delivery test data...")
    populate_housing_test(DB_PATH)

    print("  Populating income data...")
    populate_income(DB_PATH)

    print("  Populating green belt data...")
    populate_green_belt(DB_PATH)

    print("  Initialising Ibex API cache tables...")
    populate_ibex(DB_PATH)

    print("  Populating council boundary polygons...")
    populate_council_boundaries(DB_PATH)

    print("Done.")


if __name__ == "__main__":
    build_db()
