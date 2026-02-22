"""Ghost Permits section.

'Ghost permits' are planning permissions that were granted but never built out.
They indicate a council willing to approve development, but where market
conditions, viability, or developer inaction have prevented delivery.

A count above 20 is treated as a meaningful signal: the LPA is not the
bottleneck, but the market is not yet saturated — creating an opportunity
window for developers who can execute.

If the council has no ghost permit records the section is omitted.
"""
from __future__ import annotations

import pandas as pd

from report.models import CouncilContext, Insight, Metric, SectionResult
from report.sections.base import BaseSection

_SOURCE = "MHCLG / GLA Planning Application Data (approved unimplemented permissions)"
_SIGNAL_THRESHOLD = 20


class GhostPermitsSection(BaseSection):
    section_id = "ghost_permits"
    title = "Unimplemented Planning Permissions"

    @property
    def required_tables(self) -> list[str]:
        return ["ghost_permit_data"]

    def run(
        self,
        council: CouncilContext,
        data: dict[str, pd.DataFrame],
    ) -> SectionResult | None:
        df = data["ghost_permit_data"]
        council_df = df[df["council_name"] == council.council_name]

        if council_df.empty:
            return None

        count = len(council_df)

        return SectionResult(
            section_id=self.section_id,
            title=self.title,
            summary=_build_summary(council.council_name, count),
            metrics=_build_metrics(count),
            insights=_build_insights(council.council_name, count),
            data_quality="full",
            data_source=_SOURCE,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_summary(council_name: str, count: int) -> str:
    if count >= _SIGNAL_THRESHOLD:
        return (
            f"{council_name} has {count:,} planning permissions that were granted "
            f"but have not been built out. This volume signals that the local planning "
            f"authority is not the primary constraint on housing delivery — permissions "
            f"are being issued, but construction is not following. For an incoming "
            f"developer, this represents an environment where consent is achievable "
            f"and the market is not yet at saturation point."
        )
    return (
        f"{council_name} has {count:,} recorded unimplemented planning permission"
        f"{'s' if count != 1 else ''}. The low volume makes it difficult to draw "
        f"firm conclusions about LPA appetite or market dynamics from this dataset alone."
    )


def _build_metrics(count: int) -> list[Metric]:
    direction = "positive" if count >= _SIGNAL_THRESHOLD else "neutral"
    return [
        Metric(
            label="Unimplemented permissions",
            value=f"{count:,}",
            unit="",
            context=(
                "Approved but unbuilt — LPA is not the bottleneck"
                if count >= _SIGNAL_THRESHOLD
                else "Insufficient volume for a strong signal"
            ),
            direction=direction,
        ),
    ]


def _build_insights(council_name: str, count: int) -> list[Insight]:
    insights: list[Insight] = []

    if count >= _SIGNAL_THRESHOLD:
        insights.append(Insight(
            text=(
                f"With {count:,} unimplemented permissions, {council_name} demonstrates "
                f"a clear willingness to grant planning consent. The gap between approval "
                f"and delivery is most commonly explained by land-banking, viability "
                f"challenges, or developers awaiting improved market conditions — not "
                f"by planning refusals. This is a constructive signal for applicants."
            ),
            sentiment="positive",
        ))
        insights.append(Insight(
            text=(
                "The volume of unimplemented consents also suggests the market is not "
                "oversupplied: if demand were saturating the area, developers would "
                "have been incentivised to build out existing permissions. New "
                "entrants can therefore expect both a receptive LPA and room in the "
                "market to absorb additional units."
            ),
            sentiment="positive",
        ))
        if count >= 200:
            insights.append(Insight(
                text=(
                    f"The scale of unimplemented permissions ({count:,}) is particularly "
                    f"notable. It may also reflect systemic delivery constraints — such "
                    f"as infrastructure capacity, build-cost pressures, or concentrated "
                    f"land ownership — that a new developer should investigate before "
                    f"committing to a site."
                ),
                sentiment="neutral",
            ))
    else:
        insights.append(Insight(
            text=(
                f"The small number of recorded unimplemented permissions in "
                f"{council_name} could reflect strong delivery rates (permissions "
                f"being built promptly) or limited data coverage for this authority. "
                f"Cross-reference with housing delivery test results for a fuller picture."
            ),
            sentiment="neutral",
        ))

    return insights
