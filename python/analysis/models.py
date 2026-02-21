"""Core output models for the due diligence agent.

These models sit above the raw IBex API layer and above the existing
DataAnalyser, providing the structured contract consumed by the agent's
final report.  They are designed to be serialised directly to JSON for
the frontend or stored as documents.

Hierarchy:
  Enums
    TrendDirection          -- "improving" | "stable" | "declining"

  Supporting models
    ComparableApplication   -- single IBex application used as a precedent
    ConstraintFlags         -- boolean planning constraint flags for a site/area
    BoroughStats            -- approval statistics for one borough
    RiskFactor              -- a single scored risk/upside factor

  Prediction model
    ApprovalPrediction      -- 0-100 score, confidence band, ranked risk factors

  Report model
    SiteViabilityReport     -- per-borough due diligence report (one per candidate)
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from hashbrowns.ibex.models import (
    NormalisedApplicationType,
    NormalisedDecision,
    ProjectType,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrendDirection(str, Enum):
    improving = "improving"
    stable = "stable"
    declining = "declining"


# ---------------------------------------------------------------------------
# ComparableApplication
# ---------------------------------------------------------------------------


class ComparableApplication(BaseModel):
    """A single IBex planning application used as a precedent comparable.

    Carries the minimum fields needed to explain *why* a comparable was
    selected and what its outcome was.  ``similarity_score`` is set by
    the retrieval step (0–1; higher = more similar to the target).
    ``decision_weeks`` is derived from ``application_date`` / ``decided_date``
    at construction time where both are available.
    """

    model_config = ConfigDict(extra="ignore")

    planning_reference: str
    council_name: str
    url: str
    normalised_decision: NormalisedDecision
    normalised_application_type: Optional[NormalisedApplicationType] = None
    project_type: Optional[ProjectType] = None
    proposal: Optional[str] = None
    raw_address: Optional[str] = None
    application_date: Optional[date] = None
    decided_date: Optional[date] = None
    decision_weeks: Optional[float] = Field(
        default=None,
        description="Turnaround time (application → decision) in weeks.",
    )
    similarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "How similar this comparable is to the target proposal (0–1). "
            "Set by the retrieval step; 1.0 = exact match."
        ),
    )
    similarity_reasons: list[str] = Field(
        default_factory=list,
        description=(
            'Human-readable reasons this comparable was selected, '
            'e.g. ["same use class C3", "same borough", "similar scale"].'
        ),
    )


# ---------------------------------------------------------------------------
# ConstraintFlags
# ---------------------------------------------------------------------------


class ConstraintFlags(BaseModel):
    """Boolean planning constraint flags for a candidate site or borough.

    Flags are populated from a combination of IBex application metadata,
    GIS constraint layers, and heuristic inference from proposal text.
    ``data_source`` records where the flags came from so the consumer
    can calibrate trust.
    """

    model_config = ConfigDict(extra="ignore")

    # Statutory designations
    conservation_area: bool = Field(
        default=False,
        description="Site or borough contains a conservation area.",
    )
    listed_building: bool = Field(
        default=False,
        description="A listed building is present on or adjacent to the site.",
    )
    green_belt: bool = Field(
        default=False,
        description="Site falls within the Green Belt.",
    )
    aonb: bool = Field(
        default=False,
        description="Site falls within an Area of Outstanding Natural Beauty.",
    )
    sssi: bool = Field(
        default=False,
        description="Site of Special Scientific Interest designation.",
    )
    national_park: bool = Field(
        default=False,
        description="Site falls within a National Park boundary.",
    )

    # Flood and environmental
    flood_risk: bool = Field(
        default=False,
        description="Site is in Flood Zone 2 or 3 (or equivalent).",
    )
    flood_zone_3: bool = Field(
        default=False,
        description="Site is in the high-risk Flood Zone 3 specifically.",
    )

    # Planning policy overlays
    article_4: bool = Field(
        default=False,
        description="Article 4 Direction applies, removing permitted development rights.",
    )
    tree_preservation_order: bool = Field(
        default=False,
        description="One or more TPO-protected trees on or adjacent to the site.",
    )
    permitted_development_restricted: bool = Field(
        default=False,
        description="Any other restriction on permitted development rights.",
    )

    # Social / infrastructure
    community_infrastructure_levy: bool = Field(
        default=False,
        description="CIL is in force in this borough for the proposed use class.",
    )
    affordable_housing_required: bool = Field(
        default=False,
        description=(
            "Borough policy requires affordable housing contribution "
            "(typically triggered at 10+ units)."
        ),
    )

    # Provenance
    data_source: str = Field(
        default="unknown",
        description=(
            '"ibex_metadata", "gis_overlay", "heuristic", '
            '"mock", or "unknown".'
        ),
    )

    @property
    def active_flags(self) -> list[str]:
        """Return the names of all True boolean flags."""
        return [
            field
            for field, value in self.__dict__.items()
            if isinstance(value, bool) and value
        ]

    @property
    def constraint_count(self) -> int:
        """Number of active constraint flags."""
        return len(self.active_flags)


# ---------------------------------------------------------------------------
# BoroughStats
# ---------------------------------------------------------------------------


class BoroughStats(BaseModel):
    """Approval / refusal statistics for a single borough.

    Extended version of the DataAnalyser's internal BoroughStats — adds
    IBex StatsResponse fields and comparable count so the due diligence
    agent can carry richer context through to the SiteViabilityReport.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    council_id: Optional[int] = None

    # Counts
    total_applications: int = 0
    approved: int = 0
    refused: int = 0
    pending: int = 0
    withdrawn: int = 0
    comparable_count: int = Field(
        default=0,
        description="Number of comparable applications found for this borough.",
    )

    # Rates
    approval_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Approval rate as a percentage (0–100).",
    )
    refusal_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Refusal rate as a percentage (0–100).",
    )

    # Timing
    avg_decision_weeks: Optional[float] = Field(
        default=None,
        description="Average time from submission to decision, in weeks.",
    )

    # Trend
    trend: Optional[TrendDirection] = Field(
        default=None,
        description="Direction of approval rate change vs prior period.",
    )
    trend_detail: Optional[str] = Field(
        default=None,
        description='E.g. "72% approval in last 2 years vs 65% prior".',
    )

    # From IBex StatsResponse
    development_activity_level: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description="Council-level development activity classification from IBex.",
    )
    num_new_homes_approved: Optional[int] = Field(
        default=None,
        description="Total new homes approved per IBex StatsResponse.",
    )

    # Data provenance
    data_quality: Literal["full", "partial", "mock"] = Field(
        default="mock",
        description=(
            '"full" (10+ real decisions), "partial" (<10), '
            'or "mock" (no real data).'
        ),
    )


# ---------------------------------------------------------------------------
# RiskFactor
# ---------------------------------------------------------------------------


class RiskFactor(BaseModel):
    """A single factor that raises or lowers the approval prediction score.

    Positive ``score_impact`` values are bonuses; negative values are
    penalties.  The ``category`` groups related factors so the frontend
    can colour-code them.
    """

    model_config = ConfigDict(extra="ignore")

    label: str = Field(description='Short label, e.g. "Conservation area".')
    description: str = Field(
        description='One-sentence explanation, e.g. "Site is in a conservation area; '
        'listed building consent will be required in addition to full planning."',
    )
    score_impact: int = Field(
        description="Adjustment to the approval score (-100 to +100).",
        ge=-100,
        le=100,
    )
    category: str = Field(
        default="constraint",
        description=(
            '"constraint", "policy", "precedent", "design", '
            '"viability", "timing", or "other".'
        ),
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Planning reference or source supporting this factor.",
    )


# ---------------------------------------------------------------------------
# ApprovalPrediction
# ---------------------------------------------------------------------------


class ApprovalPrediction(BaseModel):
    """Predicted approval probability and supporting evidence.

    ``score`` is a 0–100 integer analogous to a probability percentage,
    derived from borough approval rates, comparable outcomes, and
    constraint penalties.  ``confidence`` reflects how much real data
    backs the estimate.
    """

    model_config = ConfigDict(extra="ignore")

    score: int = Field(
        ge=0,
        le=100,
        description=(
            "Approval probability estimate (0–100). "
            "70+ = likely approved, 40–69 = uncertain, <40 = likely refused."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the score (0–1). "
            "Low values indicate sparse comparable data."
        ),
    )

    # Supporting data
    comparable_approval_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Approval rate (%) across comparable applications used for this prediction.",
    )
    num_comparables: int = Field(
        default=0,
        description="Number of comparable applications used to derive the score.",
    )
    borough_baseline_rate: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Borough-wide approval rate (%) regardless of application type.",
    )

    # Risk and upside factors (ordered by absolute impact, descending)
    risk_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Factors that reduce the score, sorted by impact (most severe first).",
    )
    positive_factors: list[RiskFactor] = Field(
        default_factory=list,
        description="Factors that increase the score, sorted by impact (largest first).",
    )

    # Human-readable top-line
    verdict: str = Field(
        default="",
        description=(
            'One-sentence plain-English verdict, e.g. '
            '"Moderate approval likelihood — conservation area is the main risk."'
        ),
    )

    @property
    def top_risk_factors(self) -> list[str]:
        """Labels of the top 3 risk factors (descending impact)."""
        return [f.label for f in self.risk_factors[:3]]

    @property
    def band(self) -> Literal["high", "medium", "low"]:
        """Coarse approval likelihood band for UI colour coding."""
        if self.score >= 70:
            return "high"
        if self.score >= 40:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# SiteViabilityReport
# ---------------------------------------------------------------------------


class SiteViabilityReport(BaseModel):
    """Full due diligence report for a single borough candidate site.

    One report is generated per borough under consideration.  Reports are
    returned as a ranked list (``rank=1`` is the recommended borough).
    The ``viability_score`` is a composite of approval prediction, constraint
    burden, borough activity, and decision speed; it is the primary sort key.
    """

    model_config = ConfigDict(extra="ignore")

    # Identity
    borough: str = Field(description="Borough / local planning authority name.")
    council_id: Optional[int] = None
    rank: int = Field(
        ge=1,
        description="1-based rank across all candidate boroughs (1 = most viable).",
    )
    recommended: bool = Field(
        default=False,
        description="True only for rank=1 (the top recommendation).",
    )

    # Composite score
    viability_score: int = Field(
        ge=0,
        le=100,
        description=(
            "Overall site viability score (0–100). "
            "Combines approval prediction, constraint burden, borough activity, "
            "and decision speed."
        ),
    )
    viability_band: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Coarse viability band derived from viability_score.",
    )

    # Core sub-reports
    approval_prediction: ApprovalPrediction
    borough_stats: BoroughStats
    constraint_flags: ConstraintFlags

    # Comparable evidence
    comparable_applications: list[ComparableApplication] = Field(
        default_factory=list,
        description=(
            "Precedent applications used to derive the approval prediction, "
            "ordered by similarity_score descending."
        ),
    )

    # Narrative
    summary: str = Field(
        description=(
            "2–4 sentence plain-English due diligence summary for this borough, "
            "covering approval likelihood, key constraints, and decision speed."
        ),
    )
    key_considerations: list[str] = Field(
        default_factory=list,
        description=(
            "Bullet-point takeaways for the agent's report, "
            'e.g. ["Article 4 Direction applies — PD rights restricted", '
            '"Fast LPA: avg 9 weeks to decision"].'
        ),
    )

    # Data provenance
    data_quality: Literal["full", "partial", "mock"] = Field(
        default="mock",
        description='Overall data quality for this report: "full", "partial", or "mock".',
    )
    ibex_query_params: Optional[dict] = Field(
        default=None,
        description="The IBex search parameters used to retrieve comparables.",
    )

    @classmethod
    def rank_reports(cls, reports: list[SiteViabilityReport]) -> list[SiteViabilityReport]:
        """Sort reports by viability_score descending and assign ranks in-place."""
        sorted_reports = sorted(reports, key=lambda r: r.viability_score, reverse=True)
        for i, report in enumerate(sorted_reports, start=1):
            report.rank = i
            report.recommended = i == 1
        return sorted_reports
