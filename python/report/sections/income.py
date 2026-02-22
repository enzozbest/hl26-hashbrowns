"""Income Profile section.

Analyses disposable household income for the target council relative to
the national distribution and the London sub-market, and translates those
figures into planning-relevant observations about market demand, viability,
and affordable housing pressure.
"""

from __future__ import annotations

import pandas as pd

from report.models import CouncilContext, Insight, Metric, SectionResult
from report.sections.base import BaseSection

_SOURCE = "ONS Disposable Household Income Estimates (MSOA-level), aggregated to LA"


class IncomeProfileSection(BaseSection):
    section_id = "income_profile"
    title = "Household Income Profile"

    @property
    def required_tables(self) -> list[str]:
        return ["income_data"]

    def run(self, council: CouncilContext, data: dict[str, pd.DataFrame]) -> SectionResult:
        df = data["income_data"]
        year_df = df[df["year"] == council.year]

        council_row = year_df[
            year_df["ons_code"] == council.ons_code
        ]

        if council_row.empty:
            return SectionResult(
                section_id=self.section_id,
                title=self.title,
                summary="Income data is unavailable for this council.",
                data_quality="unavailable",
                data_source=_SOURCE,
            )

        council_income = council_row.iloc[0]["mean_disposable_income"]
        region_name = council_row.iloc[0]["region_name"]

        national_mean = year_df["mean_disposable_income"].mean()
        regional_df = year_df[year_df["region_name"] == region_name]
        regional_mean = regional_df["mean_disposable_income"].mean()

        is_london = region_name == "London"
        london_df = year_df[year_df["region_name"] == "London"]
        london_mean = london_df["mean_disposable_income"].mean()

        national_rank = int(
            year_df["mean_disposable_income"]
            .rank(ascending=False, method="min")[council_row.index[0]]
        )
        national_total = len(year_df)

        regional_rank = int(
            regional_df["mean_disposable_income"]
            .rank(ascending=False, method="min")[council_row.index[0]]
        )
        regional_total = len(regional_df)

        vs_national_pct = (council_income / national_mean - 1) * 100
        vs_regional_pct = (council_income / regional_mean - 1) * 100

        metrics = [
            Metric(
                label="Mean disposable income",
                value=f"{council_income:,.0f}",
                unit="£",
                context=f"Financial year ending March {council.year}",
                direction="neutral",
            ),
            Metric(
                label="National rank",
                value=f"{national_rank} / {national_total}",
                context=f"{'Higher' if national_rank <= national_total // 2 else 'Lower'} half nationally",
                direction="positive" if national_rank <= national_total // 4 else
                          "negative" if national_rank >= national_total * 3 // 4 else "neutral",
            ),
            Metric(
                label=f"{region_name} rank",
                value=f"{regional_rank} / {regional_total}",
                direction="positive" if regional_rank <= regional_total // 4 else
                          "negative" if regional_rank >= regional_total * 3 // 4 else "neutral",
            ),
            Metric(
                label="vs National average",
                value=f"{vs_national_pct:+.1f}",
                unit="%",
                context=f"National mean: £{national_mean:,.0f}",
                direction="positive" if vs_national_pct > 5 else
                          "negative" if vs_national_pct < -5 else "neutral",
            ),
            Metric(
                label=f"vs {region_name} average",
                value=f"{vs_regional_pct:+.1f}",
                unit="%",
                context=f"{region_name} mean: £{regional_mean:,.0f}",
                direction="positive" if vs_regional_pct > 5 else
                          "negative" if vs_regional_pct < -5 else "neutral",
            ),
        ]

        insights = _derive_insights(
            council_income=council_income,
            national_mean=national_mean,
            regional_mean=regional_mean,
            london_mean=london_mean if is_london else None,
            national_rank=national_rank,
            national_total=national_total,
            regional_rank=regional_rank,
            regional_total=regional_total,
            region_name=region_name,
        )

        summary = _build_summary(
            council_name=council.council_name,
            council_income=council_income,
            national_mean=national_mean,
            vs_national_pct=vs_national_pct,
            national_rank=national_rank,
            national_total=national_total,
            region_name=region_name,
        )

        return SectionResult(
            section_id=self.section_id,
            title=self.title,
            summary=summary,
            metrics=metrics,
            insights=insights,
            data_quality="full",
            data_source=_SOURCE,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summary(
    council_name: str,
    council_income: float,
    national_mean: float,
    vs_national_pct: float,
    national_rank: int,
    national_total: int,
    region_name: str,
) -> str:
    direction = "above" if vs_national_pct > 0 else "below"
    band = _income_band(council_income, national_mean)

    return (
        f"{council_name} has a mean disposable household income of £{council_income:,.0f}, "
        f"ranking {national_rank} of {national_total} local authorities nationally "
        f"({abs(vs_national_pct):.1f}% {direction} the national average of £{national_mean:,.0f}). "
        f"This positions the borough as a {band} market within the {region_name} region, "
        f"with corresponding implications for residential demand, land values, and "
        f"affordable housing requirements."
    )


def _derive_insights(
    council_income: float,
    national_mean: float,
    regional_mean: float,
    london_mean: float | None,
    national_rank: int,
    national_total: int,
    regional_rank: int,
    regional_total: int,
    region_name: str,
) -> list[Insight]:
    insights: list[Insight] = []
    national_percentile = 1 - (national_rank / national_total)

    # Market positioning
    if national_percentile >= 0.75:
        insights.append(Insight(
            text=(
                "High-income market: strong private demand for market-rate and premium "
                "residential units. Land values are likely to be elevated."
            ),
            sentiment="positive",
        ))
    elif national_percentile >= 0.50:
        insights.append(Insight(
            text=(
                "Mid-market income profile: balanced demand across market-rate and "
                "affordable tenures. Viability for standard residential schemes is "
                "generally supported."
            ),
            sentiment="neutral",
        ))
    else:
        insights.append(Insight(
            text=(
                "Lower-income market: demand skews towards affordable and social rented "
                "tenures. Private sale margins may be constrained, and viability "
                "assessments will require careful attention."
            ),
            sentiment="negative",
        ))

    # Affordable housing pressure
    if council_income < national_mean * 0.90:
        insights.append(Insight(
            text=(
                "Income levels significantly below the national average increase the "
                "political and policy pressure for affordable housing provision; "
                "expect robust Section 106 / CIL negotiations."
            ),
            sentiment="negative",
        ))
    elif council_income > national_mean * 1.20:
        insights.append(Insight(
            text=(
                "Strong income base typically correlates with higher planning authority "
                "affordable housing targets (often 35–50%); model schemes accordingly."
            ),
            sentiment="neutral",
        ))

    # Regional standing
    regional_percentile = 1 - (regional_rank / regional_total)
    if regional_percentile >= 0.75:
        insights.append(Insight(
            text=(
                f"Top quartile within the {region_name} region: this council is one of "
                f"the wealthiest local authorities in its area, supporting premium "
                f"residential values."
            ),
            sentiment="positive",
        ))
    elif regional_percentile < 0.25:
        insights.append(Insight(
            text=(
                f"Bottom quartile within the {region_name} region: relative income "
                f"disadvantage may limit achievable sales values compared to neighbouring "
                f"authorities."
            ),
            sentiment="negative",
        ))

    # London-specific context
    if london_mean is not None:
        vs_london = (council_income / london_mean - 1) * 100
        if vs_london > 10:
            insights.append(Insight(
                text=(
                    f"Income is {vs_london:.1f}% above the London average, placing this "
                    f"borough firmly in prime or near-prime market territory."
                ),
                sentiment="positive",
            ))
        elif vs_london < -10:
            insights.append(Insight(
                text=(
                    f"Income is {abs(vs_london):.1f}% below the London average; "
                    f"this borough sits in the more affordable segment of the London market, "
                    f"with corresponding regeneration and grant-funding opportunities."
                ),
                sentiment="neutral",
            ))

    return insights


def _income_band(income: float, national_mean: float) -> str:
    ratio = income / national_mean
    if ratio >= 1.30:
        return "prime"
    if ratio >= 1.10:
        return "upper mid-market"
    if ratio >= 0.90:
        return "mid-market"
    if ratio >= 0.75:
        return "lower mid-market"
    return "affordable-led"
