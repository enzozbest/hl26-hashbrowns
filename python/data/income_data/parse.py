from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent


def parse_data():
    records = []

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
            records.append((
                str(year),
                row[0],  # local_authority_code
                row[1],  # local_authority_name
                row[2],  # region_name
                row[3],  # mean_disposable_income
            ))

    return records
