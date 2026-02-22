"""Report builder — resolves a council name and runs all registered sections.

Usage::

    from report.builder import build_report

    report = build_report("Hackney")
    # or by ONS code:
    report = build_report("E09000012")
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from data import query
from report.models import CouncilContext, CouncilReport, OraclePrediction
from report.sections import SECTIONS
from report.sections.approval_prediction import ApprovalPredictionSection


def build_report(
    council: str,
    year: str = "2023",
    *,
    oracle_prediction: Optional[OraclePrediction] = None,
) -> CouncilReport:
    """Build a :class:`~report.models.CouncilReport` for the given council.

    Args:
        council:  Local authority name (case-insensitive partial match) or
                  ONS code (e.g. ``"E09000012"``).
        year:     Data vintage year.  Must match a year present in the DB.
        oracle_prediction:  Optional output from the planning-oracle neural
            network.  When provided, the *Approval Prediction* section is
            populated with ML-based approval probabilities and council
            rankings.  Pass ``None`` to omit the section gracefully.

    Returns:
        A fully populated :class:`~report.models.CouncilReport`.

    Raises:
        ValueError: If the council cannot be resolved or the year is absent.
    """
    ctx = _resolve_council(council, year)
    data = _load_data(year)

    sections = []
    for section in SECTIONS:
        if isinstance(section, ApprovalPredictionSection):
            sections.append(section.run(ctx, data, oracle=oracle_prediction))
        else:
            sections.append(section.run(ctx, data))

    return CouncilReport(council=ctx, sections=sections)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_council(council: str, year: str) -> CouncilContext:
    df = query("income_data")

    df = df[df["year"] == year]
    if df.empty:
        raise ValueError(f"No income data found for year '{year}'.")

    # Try exact ONS code match first, then case-insensitive name match.
    match = df[df["local_authority_code"].str.upper() == council.upper()]
    if match.empty:
        match = df[df["local_authority_name"].str.lower() == council.lower()]
    if match.empty:
        # Partial name match as a fallback.
        match = df[df["local_authority_name"].str.lower().str.contains(council.lower())]

    if match.empty:
        available = sorted(df["local_authority_name"].tolist())
        raise ValueError(
            f"Council '{council}' not found for year {year}. "
            f"Available: {available}"
        )
    if len(match) > 1:
        names = match["local_authority_name"].tolist()
        raise ValueError(
            f"Ambiguous council name '{council}' — matched: {names}. "
            f"Please be more specific or use the ONS code."
        )

    row = match.iloc[0]
    return CouncilContext(
        local_authority_code=row["local_authority_code"],
        local_authority_name=row["local_authority_name"],
        region_name=row["region_name"],
        year=year,
    )


def _load_data(year: str) -> dict[str, pd.DataFrame]:
    """Load all tables required by any registered section, keyed by table name."""
    required = {table for section in SECTIONS for table in section.required_tables}
    return {table: query(table) for table in required}
