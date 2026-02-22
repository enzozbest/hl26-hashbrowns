"""Green Belt Coverage section.

Computes the proportion of a council's area that falls within designated
green belt land, using the overlap between the council's boundary polygon
and the DLUHC green belt dataset.  Projects both layers to British National
Grid (EPSG:27700) for accurate area calculations.

If the council has no boundary polygon in the database the section is
omitted from the report entirely (returns None).
"""

from __future__ import annotations

import pandas as pd
from shapely import wkb
from shapely.ops import transform, unary_union
import pyproj

from report.models import CouncilContext, Insight, Metric, SectionResult
from report.sections.base import BaseSection

_SOURCE = "DLUHC Green Belt boundaries; ONS LAD boundary polygons (martinjc/UK-GeoJSON)"

# Reusable transformer WGS84 → British National Grid
_TO_BNG = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:27700", always_xy=True
).transform


def _to_bng(geom):
    return transform(_TO_BNG, geom)


class GreenBeltSection(BaseSection):
    section_id = "green_belt"
    title = "Green Belt Coverage"

    @property
    def required_tables(self) -> list[str]:
        return ["council_boundaries", "green_belt"]

    def run(
        self,
        council: CouncilContext,
        data: dict[str, pd.DataFrame],
    ) -> SectionResult | None:
        boundaries_df = data["council_boundaries"]
        green_belt_df = data["green_belt"]

        # Locate council boundary polygon — if absent, omit section entirely
        boundary_row = boundaries_df[boundaries_df["ons_code"] == council.ons_code]
        if boundary_row.empty:
            return None

        council_geom_wgs = wkb.loads(bytes(boundary_row["geometry"].iloc[0]))

        # Load and union all green belt polygons (active entries only)
        gb_geoms = []
        for raw in green_belt_df["geometry"]:
            if raw is None:
                continue
            try:
                gb_geoms.append(wkb.loads(bytes(raw)))
            except Exception:
                continue

        if not gb_geoms:
            return None

        # Project both layers to BNG for metric-accurate area (m²)
        council_bng = _to_bng(council_geom_wgs)
        gb_union_bng = _to_bng(unary_union(gb_geoms))

        council_area_m2 = council_bng.area
        intersection = council_bng.intersection(gb_union_bng)
        intersection_area_m2 = intersection.area

        pct = (intersection_area_m2 / council_area_m2) * 100 if council_area_m2 > 0 else 0.0

        council_area_km2 = council_area_m2 / 1_000_000
        intersection_area_km2 = intersection_area_m2 / 1_000_000
        developable_km2 = council_area_km2 - intersection_area_km2

        has_green_belt = pct > 0.05  # treat trace overlaps as zero

        metrics = _build_metrics(pct, council_area_km2, intersection_area_km2, developable_km2, has_green_belt)
        insights = _build_insights(pct, council.council_name, has_green_belt)
        summary = _build_summary(council.council_name, pct, council_area_km2, intersection_area_km2, has_green_belt)

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


def _build_metrics(
    pct: float,
    council_area_km2: float,
    gb_area_km2: float,
    developable_km2: float,
    has_green_belt: bool,
) -> list[Metric]:
    if not has_green_belt:
        return [
            Metric(
                label="Green belt coverage",
                value="0",
                unit="%",
                context="No designated green belt land",
                direction="positive",
            ),
            Metric(
                label="Council area",
                value=f"{council_area_km2:.1f}",
                unit=" km²",
                direction="neutral",
            ),
        ]

    direction = (
        "negative" if pct >= 50
        else "neutral" if pct >= 15
        else "positive"
    )
    return [
        Metric(
            label="Green belt coverage",
            value=f"{pct:.1f}",
            unit="%",
            context=_coverage_band_label(pct),
            direction=direction,
        ),
        Metric(
            label="Green belt area",
            value=f"{gb_area_km2:.1f}",
            unit=" km²",
            direction="neutral",
        ),
        Metric(
            label="Council area",
            value=f"{council_area_km2:.1f}",
            unit=" km²",
            direction="neutral",
        ),
        Metric(
            label="Non-green-belt area",
            value=f"{developable_km2:.1f}",
            unit=" km²",
            context="Approximate unconstrained land mass",
            direction="positive",
        ),
    ]


def _build_insights(pct: float, council_name: str, has_green_belt: bool) -> list[Insight]:
    insights: list[Insight] = []

    if not has_green_belt:
        insights.append(Insight(
            text=(
                f"{council_name} contains no designated green belt land. "
                f"Development proposals are not subject to the strong presumption "
                f"against development in the green belt, which typically simplifies "
                f"the planning consent pathway for residential and commercial schemes."
            ),
            sentiment="positive",
        ))
        insights.append(Insight(
            text=(
                "The absence of green belt does not remove other planning constraints "
                "(AONB, flood risk, heritage assets, local plan policies). Independent "
                "site-level assessments remain essential."
            ),
            sentiment="neutral",
        ))
        return insights

    if pct >= 75:
        insights.append(Insight(
            text=(
                f"Green belt covers a large majority ({pct:.0f}%) of {council_name}. "
                f"Residential and commercial development across most of the borough "
                f"faces the high planning bar of 'very special circumstances' or must "
                f"demonstrate exceptional need — materially elevating consent risk."
            ),
            sentiment="negative",
        ))
    elif pct >= 50:
        insights.append(Insight(
            text=(
                f"Over half of {council_name} ({pct:.0f}%) is designated green belt. "
                f"Development potential is predominantly concentrated in existing "
                f"settlements and brownfield land outside the green belt envelope."
            ),
            sentiment="negative",
        ))
    elif pct >= 15:
        insights.append(Insight(
            text=(
                f"Green belt accounts for {pct:.0f}% of {council_name}. "
                f"A meaningful portion of the borough is subject to green belt "
                f"restrictions; site selection should prioritise land outside this "
                f"designation to reduce policy risk."
            ),
            sentiment="neutral",
        ))
    else:
        insights.append(Insight(
            text=(
                f"Green belt covers a relatively small proportion ({pct:.0f}%) of "
                f"{council_name}. The constraint is present but geographically limited; "
                f"the majority of the borough remains outside the protected designation."
            ),
            sentiment="neutral",
        ))

    insights.append(Insight(
        text=(
            "Green belt boundaries are subject to periodic Local Plan reviews. "
            "Proposed development abutting or within the green belt should be "
            "assessed against NPPF Chapter 13 policies and any emerging plan "
            "changes that may alter the designated envelope."
        ),
        sentiment="neutral",
    ))

    if pct >= 30:
        insights.append(Insight(
            text=(
                "Brownfield land within existing settlement boundaries and "
                "previously developed sites are typically the most viable "
                "development opportunities in high-green-belt authorities. "
                "Exception sites and affordable rural housing schemes may also "
                "provide limited pathways through the green belt designation."
            ),
            sentiment="positive",
        ))

    return insights


def _build_summary(
    council_name: str,
    pct: float,
    council_area_km2: float,
    gb_area_km2: float,
    has_green_belt: bool,
) -> str:
    if not has_green_belt:
        return (
            f"{council_name} does not contain any designated green belt land. "
            f"With a total area of {council_area_km2:.1f} km², the council is entirely "
            f"outside the green belt designation, meaning development proposals are "
            f"not subject to the strong NPPF presumption against green belt development. "
            f"This represents a materially lower policy constraint than comparable "
            f"authorities with green belt coverage."
        )

    band = _coverage_band_label(pct)
    return (
        f"Approximately {pct:.1f}% of {council_name} ({gb_area_km2:.1f} km² of a "
        f"{council_area_km2:.1f} km² total) falls within the designated green belt. "
        f"This places the council in the '{band}' category. Green belt policy under "
        f"the NPPF strongly restricts new development within the designated area; "
        f"proposals must demonstrate very special circumstances or fall within "
        f"defined exceptions. The unconstrained land mass should guide site "
        f"selection and viability modelling."
    )


def _coverage_band_label(pct: float) -> str:
    if pct >= 75:
        return "very high green belt constraint"
    if pct >= 50:
        return "high green belt constraint"
    if pct >= 15:
        return "moderate green belt constraint"
    return "low green belt constraint"
