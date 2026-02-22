"""Domain models for the council-level planning report.

Hierarchy:
    CouncilContext      -- resolved identity of the council under analysis
    Metric              -- a single labelled data point with optional context
    Insight             -- a plain-English observation derived from data
    SectionResult       -- the output of one analytical section
    CouncilReport       -- the fully assembled report across all sections
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# CouncilContext
# ---------------------------------------------------------------------------


class CouncilContext(BaseModel):
    """Resolved identity of the council being analysed."""

    model_config = ConfigDict(extra="ignore")

    local_authority_code: str
    local_authority_name: str
    region_name: str
    year: str = Field(description="Data vintage year, e.g. '2023'.")


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


class Metric(BaseModel):
    """A single labelled data point for display in the report."""

    model_config = ConfigDict(extra="ignore")

    label: str = Field(description="Short human-readable label, e.g. 'Mean income'.")
    value: float | int | str = Field(description="The metric value.")
    unit: Optional[str] = Field(
        default=None,
        description="Optional unit string appended on display, e.g. '£', '%', 'weeks'.",
    )
    context: Optional[str] = Field(
        default=None,
        description="One-line comparative context, e.g. '12% above national average'.",
    )
    direction: Optional[Literal["positive", "negative", "neutral"]] = Field(
        default="neutral",
        description="Semantic direction used for colour-coding in the renderer.",
    )


# ---------------------------------------------------------------------------
# Insight
# ---------------------------------------------------------------------------


class Insight(BaseModel):
    """A plain-English observation derived from one or more metrics."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(description="The observation as a complete sentence.")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        default="neutral",
        description="Used by the renderer to colour-code bullet points.",
    )


# ---------------------------------------------------------------------------
# SectionResult
# ---------------------------------------------------------------------------


class SectionResult(BaseModel):
    """The fully computed output of a single analytical section."""

    model_config = ConfigDict(extra="ignore")

    section_id: str = Field(
        description="Stable identifier for this section, e.g. 'income_profile'."
    )
    title: str = Field(description="Display title shown as the section heading.")
    summary: str = Field(
        description="2–3 sentence plain-English narrative for this section."
    )
    metrics: list[Metric] = Field(
        default_factory=list,
        description="Key-figure metrics displayed in a summary grid.",
    )
    insights: list[Insight] = Field(
        default_factory=list,
        description="Bullet-point observations derived from the data.",
    )
    data_quality: Literal["full", "partial", "unavailable"] = Field(
        default="full",
        description=(
            "'full' — complete data available; "
            "'partial' — incomplete data, treat with caution; "
            "'unavailable' — section could not be computed."
        ),
    )
    data_source: str = Field(
        default="",
        description="Human-readable attribution, e.g. 'ONS MSOA income estimates 2023'.",
    )


# ---------------------------------------------------------------------------
# CouncilReport
# ---------------------------------------------------------------------------


class CouncilReport(BaseModel):
    """Fully assembled planning intelligence report for a single council."""

    model_config = ConfigDict(extra="ignore")

    council: CouncilContext
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sections: list[SectionResult] = Field(
        default_factory=list,
        description="Ordered list of analytical sections.",
    )

    @property
    def section_by_id(self) -> dict[str, SectionResult]:
        return {s.section_id: s for s in self.sections}
