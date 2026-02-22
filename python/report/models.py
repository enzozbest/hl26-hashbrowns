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
from typing import Any, Literal, Optional

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
# OraclePrediction — neural-network output passed into the report
# ---------------------------------------------------------------------------


class IndicatorEntry(BaseModel):
    """A single feature indicator that influenced a council's score."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Feature name, e.g. 'approval_rate'.")
    value: float = Field(description="Raw feature value used by the model.")
    contribution: float = Field(
        description="Signed magnitude of this feature's contribution.",
    )
    direction: str = Field(description="'positive' or 'negative' contributor.")


class CouncilPrediction(BaseModel):
    """A single council's prediction from the planning oracle."""

    model_config = ConfigDict(extra="allow")

    council_id: int
    council_name: Optional[str] = None
    score: float = Field(description="Approval affinity score (0-1).")
    indicators: list[IndicatorEntry] = Field(
        default_factory=list,
        description="Ranked features that drove this borough's score.",
    )


class OraclePrediction(BaseModel):
    """Output from the planning-oracle neural network.

    Passed into the report builder so that sections can incorporate
    ML-based approval predictions alongside traditional data analysis.

    The ``reasonings`` field is intentionally loose (list of free-form
    dicts) because the neural network does not yet produce structured
    reasoning output.  When it does, consumers should adapt the
    ``format_reasonings`` helper rather than changing the field type.
    """

    model_config = ConfigDict(extra="allow")

    approval_probability: float = Field(
        description="Calibrated approval probability (0-1).",
    )
    confidence_interval: tuple[float, float] = Field(
        description="Approximate 95% confidence interval (lower, upper).",
    )
    top_councils: list[CouncilPrediction] = Field(
        default_factory=list,
        description="Councils ranked by approval affinity, highest first.",
    )
    reasonings: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Free-form reasoning entries from the neural network. "
            "Each dict may contain 'text', 'factor', 'weight', or any "
            "schema the model produces. None until the model supports it."
        ),
    )


def format_reasonings(reasonings: list[dict[str, Any]]) -> list[str]:
    """Convert raw reasoning dicts into human-readable strings.

    This helper is intentionally loose so it can be adapted when the
    neural network starts producing structured reasoning output.
    Currently handles two formats:

    * ``{"text": "..."}`` — plain text reasoning
    * ``{"factor": "...", "weight": 0.3, "direction": "positive"}``
      — weighted factor

    Unknown dict shapes are serialised as-is.
    """
    lines: list[str] = []
    for entry in reasonings:
        if "text" in entry:
            lines.append(str(entry["text"]))
        elif "factor" in entry:
            weight = entry.get("weight", "")
            direction = entry.get("direction", "")
            prefix = f"[{direction}]" if direction else ""
            suffix = f" (weight: {weight})" if weight else ""
            lines.append(f"{prefix} {entry['factor']}{suffix}".strip())
        else:
            lines.append(str(entry))
    return lines


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
