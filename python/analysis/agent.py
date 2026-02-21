"""Due diligence agent — end-to-end pipeline from natural language to ranked reports.

Takes a plain-English development brief, parses it into a
:class:`~intent_parser.schema.ParsedIntent`, fans out IBex queries across
all candidate boroughs concurrently, runs the scorer and comparables finder
for each borough, and returns a ranked list of
:class:`~analysis.models.SiteViabilityReport`.

Usage::

    from analysis.agent import DueDiligenceAgent

    agent = DueDiligenceAgent(settings)
    async with agent:
        reports = await agent.run("20-unit affordable housing in South London")
        for r in reports:
            print(r.borough, r.viability_score, r.viability_band)

The agent can also accept a pre-parsed ``ParsedIntent`` directly via
:meth:`run_from_intent` to skip the parsing step (useful when the frontend
has already called ``/api/parse``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta
from typing import Optional

from hashbrowns.config import Settings
from hashbrowns.ibex.client import IbexClient
from hashbrowns.ibex.models import (
    BaseApplicationsSchema,
    NormalisedDecision,
    SearchResponse,
)
from intent_parser.schema import ParsedIntent

from .comparables import ComparableFinder
from .models import (
    ApprovalPrediction,
    RiskFactor,
    SiteViabilityReport,
)
from .scorer import BoroughScorer, ScoringResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default IBex search radius in metres.
DEFAULT_SEARCH_RADIUS_M: int = 1000

#: Date range used for IBex /applications queries (3 years back).
LOOKBACK_DAYS: int = 3 * 365

#: Weights for the composite viability score (must sum to 1.0).
VIABILITY_WEIGHTS: dict[str, float] = {
    "approval_prediction": 0.50,
    "comparable_evidence":  0.20,
    "decision_speed":       0.15,
    "borough_activity":     0.15,
}

__all__ = ["DueDiligenceAgent"]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DueDiligenceAgent:
    """Orchestrates the full due diligence pipeline.

    Owns an :class:`~hashbrowns.ibex.client.IbexClient` for API access,
    a :class:`~analysis.scorer.BoroughScorer` for statistics, and a
    :class:`~analysis.comparables.ComparableFinder` for precedent selection.

    Use as an async context manager so that the underlying HTTP client is
    properly closed::

        async with DueDiligenceAgent(settings) as agent:
            reports = await agent.run("query text")
    """

    def __init__(
        self,
        settings: Settings,
        *,
        parse_fn: Optional[object] = None,
        allow_mock: bool = True,
    ) -> None:
        self._settings = settings
        self._client = IbexClient(settings)
        self._scorer = BoroughScorer()
        self._finder = ComparableFinder()
        self._parse_fn = parse_fn
        self._allow_mock = allow_mock

    async def __aenter__(self) -> "DueDiligenceAgent":
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self._client.__aexit__(*args)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        *,
        top_n_comparables: int = 5,
    ) -> list[SiteViabilityReport]:
        """Full pipeline: parse → fetch → score → rank → report.

        Args:
            query:              Natural language development brief.
            top_n_comparables:  Max comparables per borough.

        Returns:
            Ranked list of :class:`SiteViabilityReport` (rank 1 = best).

        Raises:
            ValueError: If no ``parse_fn`` was provided and ``run`` is
                called (use ``run_from_intent`` instead, or inject a
                parse function).
        """
        if self._parse_fn is None:
            raise ValueError(
                "No parse function configured — use run_from_intent() "
                "or pass parse_fn to the constructor."
            )
        intent: ParsedIntent = await self._parse_fn(query)  # type: ignore[misc]
        return await self.run_from_intent(
            intent, top_n_comparables=top_n_comparables
        )

    async def run_from_intent(
        self,
        intent: ParsedIntent,
        *,
        top_n_comparables: int = 5,
    ) -> list[SiteViabilityReport]:
        """Run the pipeline from a pre-parsed intent.

        Fans out IBex queries for every resolved council concurrently,
        scores each borough, selects comparables, and builds ranked
        :class:`SiteViabilityReport` objects.
        """
        councils = intent.location.resolved_councils
        if not councils:
            logger.warning("No resolved councils — returning empty report list")
            return []

        t0 = time.monotonic()

        # Fan out per-borough work concurrently
        tasks = [
            self._process_borough(council, intent, top_n_comparables)
            for council in councils
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reports: list[SiteViabilityReport] = []
        for council, result in zip(councils, results):
            if isinstance(result, BaseException):
                logger.error("Borough %s failed: %s", council, result)
                continue
            reports.append(result)

        elapsed_ms = round((time.monotonic() - t0) * 1000)
        logger.info(
            "Due diligence complete: %d boroughs, %dms", len(reports), elapsed_ms
        )

        return SiteViabilityReport.rank_reports(reports)

    # ------------------------------------------------------------------
    # Per-borough pipeline
    # ------------------------------------------------------------------

    async def _process_borough(
        self,
        council_name: str,
        intent: ParsedIntent,
        top_n_comparables: int,
    ) -> SiteViabilityReport:
        """Fetch, score, find comparables, and build a report for one borough."""
        # 1. Fetch applications from IBex
        applications = await self._fetch_applications(council_name, intent)

        # 2. Infer constraints from the application data
        constraint_flags = self._scorer.infer_constraints(applications)

        # 3. Score the borough
        scoring: ScoringResult = self._scorer.score(
            applications,
            council_name,
            constraint_flags,
            allow_mock=self._allow_mock,
        )

        # 4. Find comparables
        comparables = self._finder.find(intent, applications, top_n=top_n_comparables)

        # 5. Build approval prediction
        prediction = self._build_prediction(scoring, comparables, intent)

        # 6. Compute composite viability score
        viability_score = self._compute_viability_score(scoring, prediction, comparables)

        # 7. Build the report
        if viability_score >= 70:
            band = "high"
        elif viability_score >= 40:
            band = "medium"
        else:
            band = "low"

        summary = self._build_summary(
            council_name, scoring, prediction, comparables
        )
        considerations = self._build_considerations(
            scoring, prediction, constraint_flags
        )

        # Update comparable count on stats
        scoring.stats.comparable_count = len(comparables)

        return SiteViabilityReport(
            borough=council_name,
            council_id=scoring.stats.council_id,
            rank=1,  # placeholder — ranked by rank_reports()
            viability_score=viability_score,
            viability_band=band,
            approval_prediction=prediction,
            borough_stats=scoring.stats,
            constraint_flags=constraint_flags,
            comparable_applications=comparables,
            summary=summary,
            key_considerations=considerations,
            data_quality=scoring.stats.data_quality,
        )

    # ------------------------------------------------------------------
    # IBex data fetching
    # ------------------------------------------------------------------

    async def _fetch_applications(
        self,
        council_name: str,
        intent: ParsedIntent,
    ) -> list[BaseApplicationsSchema]:
        """Fetch IBex applications for a borough via coordinate search.

        Uses the intent's resolved coordinates if available, otherwise
        looks up council centre coordinates from the location module.
        """
        from intent_parser.location import UK_COUNCILS

        coords = intent.location.resolved_coordinates
        if coords is None:
            council_data = UK_COUNCILS.get(council_name)
            if council_data is None:
                logger.warning("No coordinates for %s", council_name)
                return []
            coords = {"lat": council_data["lat"], "lng": council_data["lng"]}

        radius = intent.location.radius_suggestion_m or DEFAULT_SEARCH_RADIUS_M

        # Convert WGS84 to approximate OSGB36 for IBex
        easting, northing = _wgs84_to_osgb36(coords["lat"], coords["lng"])

        today = date.today()
        date_from = (today - timedelta(days=LOOKBACK_DAYS)).isoformat()
        date_to = today.isoformat()

        try:
            results: list[SearchResponse] = await self._client.search(
                coordinates=[easting, northing],
                radius=radius,
                srid=27700,
                date_from=date_from,
                date_to=date_to,
                date_range_type="any",
            )
            logger.info(
                "Fetched %d applications for %s", len(results), council_name
            )
            return results  # SearchResponse extends BaseApplicationsSchema
        except Exception as exc:
            logger.error("IBex fetch failed for %s: %s", council_name, exc)
            return []

    # ------------------------------------------------------------------
    # Approval prediction
    # ------------------------------------------------------------------

    def _build_prediction(
        self,
        scoring: ScoringResult,
        comparables: list,
        intent: ParsedIntent,
    ) -> ApprovalPrediction:
        """Build an ApprovalPrediction from scorer output and comparables."""
        # Comparable approval rate
        decided_comps = [
            c for c in comparables
            if c.normalised_decision in (NormalisedDecision.Approved, NormalisedDecision.Refused)
        ]
        approved_comps = [
            c for c in decided_comps
            if c.normalised_decision == NormalisedDecision.Approved
        ]
        comp_rate: Optional[float] = None
        if decided_comps:
            comp_rate = round(len(approved_comps) / len(decided_comps) * 100, 1)

        # Score = base approval score adjusted by comparable evidence
        score = scoring.base_approval_score
        if comp_rate is not None and decided_comps:
            # Blend base score with comparable rate (weighted by count)
            weight = min(len(decided_comps) / 10, 0.5)  # up to 50% influence
            score = round(score * (1 - weight) + comp_rate * weight)
        score = max(0, min(100, score))

        # Confidence based on data quality and comparable count
        if scoring.stats.data_quality == "full" and len(comparables) >= 3:
            confidence = 0.8
        elif scoring.stats.data_quality == "partial" or len(comparables) >= 1:
            confidence = 0.5
        else:
            confidence = 0.2

        # Build risk and positive factors
        risk_factors: list[RiskFactor] = []
        positive_factors: list[RiskFactor] = []

        for label, penalty in scoring.applied_penalties:
            risk_factors.append(RiskFactor(
                label=label,
                description=f"{label} constraint applies — may require additional consent.",
                score_impact=-penalty,
                category="constraint",
            ))

        if scoring.stats.trend is not None:
            from .models import TrendDirection
            if scoring.stats.trend == TrendDirection.declining:
                risk_factors.append(RiskFactor(
                    label="Declining approval trend",
                    description=scoring.stats.trend_detail or "Approval rates trending downward.",
                    score_impact=-5,
                    category="policy",
                ))
            elif scoring.stats.trend == TrendDirection.improving:
                positive_factors.append(RiskFactor(
                    label="Improving approval trend",
                    description=scoring.stats.trend_detail or "Approval rates trending upward.",
                    score_impact=5,
                    category="policy",
                ))

        if comp_rate is not None and comp_rate >= 70:
            positive_factors.append(RiskFactor(
                label="Strong comparable precedent",
                description=f"{comp_rate:.0f}% of similar applications were approved.",
                score_impact=10,
                category="precedent",
            ))
        elif comp_rate is not None and comp_rate < 40:
            risk_factors.append(RiskFactor(
                label="Weak comparable precedent",
                description=f"Only {comp_rate:.0f}% of similar applications were approved.",
                score_impact=-10,
                category="precedent",
            ))

        if scoring.stats.avg_decision_weeks is not None and scoring.stats.avg_decision_weeks <= 12:
            positive_factors.append(RiskFactor(
                label="Fast decision turnaround",
                description=f"Average {scoring.stats.avg_decision_weeks:.0f} weeks to decision.",
                score_impact=3,
                category="timing",
            ))

        # Sort by absolute impact
        risk_factors.sort(key=lambda f: abs(f.score_impact), reverse=True)
        positive_factors.sort(key=lambda f: abs(f.score_impact), reverse=True)

        # Verdict
        if score >= 70:
            verdict = "High approval likelihood"
        elif score >= 40:
            verdict = "Moderate approval likelihood"
        else:
            verdict = "Low approval likelihood"

        if risk_factors:
            verdict += f" — {risk_factors[0].label.lower()} is the main risk."
        else:
            verdict += "."

        return ApprovalPrediction(
            score=score,
            confidence=confidence,
            comparable_approval_rate=comp_rate,
            num_comparables=len(comparables),
            borough_baseline_rate=scoring.stats.approval_rate,
            risk_factors=risk_factors,
            positive_factors=positive_factors,
            verdict=verdict,
        )

    # ------------------------------------------------------------------
    # Composite viability score
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_viability_score(
        scoring: ScoringResult,
        prediction: ApprovalPrediction,
        comparables: list,
    ) -> int:
        """Weighted composite of approval prediction, comparables, speed, activity."""
        # Approval prediction component (0-100)
        approval_component = prediction.score

        # Comparable evidence component (0-100)
        if comparables:
            comp_component = min(100, sum(c.similarity_score for c in comparables) / len(comparables) * 100)
        else:
            comp_component = 30  # weak baseline if no comparables

        # Decision speed component (0-100): faster = better
        avg_weeks = scoring.stats.avg_decision_weeks
        if avg_weeks is not None:
            speed_component = max(0, min(100, 100 - (avg_weeks - 8) * 5))
        else:
            speed_component = 50

        # Borough activity component (0-100): more applications = more active = better
        total = scoring.stats.total_applications
        if total >= 100:
            activity_component = 100
        elif total > 0:
            activity_component = min(100, total)
        else:
            activity_component = 20

        raw = (
            VIABILITY_WEIGHTS["approval_prediction"] * approval_component
            + VIABILITY_WEIGHTS["comparable_evidence"] * comp_component
            + VIABILITY_WEIGHTS["decision_speed"] * speed_component
            + VIABILITY_WEIGHTS["borough_activity"] * activity_component
        )
        return max(0, min(100, round(raw)))

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        council_name: str,
        scoring: ScoringResult,
        prediction: ApprovalPrediction,
        comparables: list,
    ) -> str:
        """Build a 2-4 sentence plain-English summary."""
        parts: list[str] = []

        parts.append(
            f"{council_name} has a {scoring.stats.approval_rate:.0f}% approval rate "
            f"across {scoring.stats.total_applications} applications."
        )

        if prediction.score >= 70:
            parts.append(
                f"Approval likelihood is high (score: {prediction.score}/100)."
            )
        elif prediction.score >= 40:
            parts.append(
                f"Approval likelihood is moderate (score: {prediction.score}/100)."
            )
        else:
            parts.append(
                f"Approval likelihood is low (score: {prediction.score}/100)."
            )

        if scoring.applied_penalties:
            labels = [label for label, _ in scoring.applied_penalties]
            parts.append(f"Key constraints: {', '.join(labels)}.")

        if scoring.stats.avg_decision_weeks is not None:
            parts.append(
                f"Average decision time: {scoring.stats.avg_decision_weeks:.0f} weeks."
            )

        return " ".join(parts)

    @staticmethod
    def _build_considerations(
        scoring: ScoringResult,
        prediction: ApprovalPrediction,
        constraint_flags,
    ) -> list[str]:
        """Build bullet-point key considerations."""
        items: list[str] = []

        for label, penalty in scoring.applied_penalties:
            items.append(f"{label} applies (-{penalty} pts)")

        if prediction.comparable_approval_rate is not None:
            items.append(
                f"Comparable approval rate: {prediction.comparable_approval_rate:.0f}% "
                f"({prediction.num_comparables} precedents)"
            )

        if scoring.stats.trend_detail:
            items.append(scoring.stats.trend_detail)

        if scoring.stats.avg_decision_weeks is not None:
            if scoring.stats.avg_decision_weeks <= 12:
                items.append(
                    f"Fast LPA: avg {scoring.stats.avg_decision_weeks:.0f} weeks to decision"
                )
            elif scoring.stats.avg_decision_weeks >= 20:
                items.append(
                    f"Slow LPA: avg {scoring.stats.avg_decision_weeks:.0f} weeks to decision"
                )

        active = constraint_flags.active_flags
        if len(active) > len(scoring.applied_penalties):
            extra = [
                f.replace("_", " ").title()
                for f in active
                if not any(f in attr for attr, _ in scoring.applied_penalties)
            ]
            if extra:
                items.append(f"Other constraints: {', '.join(extra)}")

        return items


# ---------------------------------------------------------------------------
# Coordinate conversion helper
# ---------------------------------------------------------------------------


def _wgs84_to_osgb36(lat: float, lng: float) -> tuple[float, float]:
    """Convert WGS84 lat/lng to approximate OSGB36 easting/northing.

    Linear approximation centred on London — sufficient for search radius
    queries.  For sub-metre accuracy use ``pyproj``.
    """
    easting = 500_000 + (lng + 0.1) * 70_000
    northing = 180_000 + (lat - 51.5) * 111_000
    return (round(easting, 1), round(northing, 1))
