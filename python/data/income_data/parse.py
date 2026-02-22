from pathlib import Path

import pandas as pd

from data.councils import resolve

DATA_DIR = Path(__file__).resolve().parent


def parse_data():
    records = []
    unmatched: set[str] = set()

    for path in sorted(DATA_DIR.glob("*.xlsx")):
        year = path.stem  # e.g. "2023"
        df = pd.read_excel(path, sheet_name="Net annual income", header=3)

        aggregated = (
            df.groupby(
                ["Local authority code", "Local authority name"],
                as_index=False,
            )
            .agg(
                region_name=("Region name", "first"),
                mean_disposable_income=("Disposable (net) annual income (£)", "mean"),
            )
        )
        aggregated["mean_disposable_income"] = aggregated["mean_disposable_income"].round(2)

        for row in aggregated.itertuples(index=False):
            la_name = row[1]
            council = resolve(la_name)
            if council is None:
                unmatched.add(la_name)
                continue

            records.append((
                str(year),
                row[0],                # local_authority_code
                council.name,          # normalised local_authority_name
                council.id,            # council_id
                row[2],                # region_name
                row[3],                # mean_disposable_income
            ))

    if unmatched:
        print(f"  [income_data] skipped {len(unmatched)} unmatched authority(ies): "
              f"{sorted(unmatched)}")

    return records
