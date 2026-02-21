"""Borough scorer — computes approval statistics and a constraint-penalised base score.

Sits between raw IBex application data and the due diligence agent's
``SiteViabilityReport``.  ``BoroughScorer`` works in two modes:

- **Real mode** — a non-empty list of typed ``BaseApplicationsSchema`` objects
  (``SearchResponse`` or ``ApplicationsResponse``).  All computation is
  derived from the actual application records.
- **Mock mode** — triggered when ``applications`` is empty.  Generates
  deterministic, plausible statistics seeded on the borough name so the
  frontend can render while real data is being fetched.

Penalty modifiers (applied to the raw approval rate to produce
``base_approval_score``):

=================== =======
Constraint          Penalty
=================== =======
Flood zone          -15 pts
Conservation area   -20 pts
Green Belt          -25 pts
Article 4           -10 pts
=================== =======

Usage::

    from hashbrowns.ibex.models import SearchResponse
    from analysis.models import ConstraintFlags
    from analysis.scorer import BoroughScorer

    scorer = BoroughScorer()
    flags  = ConstraintFlags(conservation_area=True, article_4=True)
    result = scorer.score(applications, "Camden", flags, council_id=240)

    result.stats.approval_rate      # 71.2  (raw borough rate from data)
    result.base_approval_score      # 41    (71 - 20 conservation - 10 article 4)
    result.approval_by_type         # {"full planning application": 68.5, ...}
    result.applied_penalties        # [("Conservation area", 20), ("Article 4", 10)]
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from hashbrowns.ibex.models import (
    BaseApplicationsSchema,
    NormalisedApplicationType,
    NormalisedDecision,
)

from .models import BoroughStats, ConstraintFlags, TrendDirection


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Constraint → (display label, penalty deducted from approval rate).
CONSTRAINT_PENALTIES: dict[str, tuple[str, int]] = {
    "flood_risk":       ("Flood zone",        15),
    "conservation_area":("Conservation area", 20),
    "green_belt":       ("Green Belt",        25),
    "article_4":        ("Article 4",         10),
}

#: Decided applications needed for "full" data quality (vs "partial").
FULL_QUALITY_THRESHOLD: int = 10

#: Trend window — compare last N days against everything prior.
TREND_WINDOW_DAYS: int = 2 * 365

#: Minimum decided applications per window for trend to be reported.
TREND_MIN_SAMPLES: int = 3

#: Maximum plausible submission → decision gap (guards against bad data).
MAX_DECISION_DAYS: int = 3 * 365

#: Decisions that count toward approval/refusal rate.
_DECIDED: frozenset[NormalisedDecision] = frozenset(
    {NormalisedDecision.Approved, NormalisedDecision.Refused}
)

#: Application types that signal a conservation area overlay.
_CONSERVATION_AREA_TYPES: frozenset[NormalisedApplicationType] = frozenset(
    {
        NormalisedApplicationType.conservation_area,
        NormalisedApplicationType.listed_building_consent,
    }
)

__all__ = ["BoroughScorer", "ScoringResult", "CONSTRAINT_PENALTIES"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ScoringResult:
    """Output of :meth:`BoroughScorer.score`.

    ``stats`` holds the raw borough statistics.
    ``base_approval_score`` is the constraint-penalised starting point for
    ``ApprovalPrediction.score`` (0–100 integer).
    """

    stats: BoroughStats
    base_approval_score: int
    """Approval rate minus constraint penalties, clamped to 0–100."""

    approval_by_type: dict[str, float]
    """``normalised_application_type`` value → approval rate percentage."""

    applied_penalties: list[tuple[str, int]] = field(default_factory=list)
    """Each ``(label, points_deducted)`` pair that was applied."""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class BoroughScorer:
    """Scores a list of IBex applications for a single borough.

    Stateless — safe to create once and call :meth:`score` for multiple
    boroughs.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        applications: list[BaseApplicationsSchema],
        borough_name: str,
        constraint_flags: ConstraintFlags,
        council_id: Optional[int] = None,
        *,
        allow_mock: bool = True,          # <-- new flag
    ) -> ScoringResult:
        """Compute :class:`BoroughStats` and a penalised base approval score.

        Falls back to :meth:`_mock_score` when *applications* is empty.

        Args:
            applications:     Typed IBex records for this borough.  Both
                              ``SearchResponse`` and ``ApplicationsResponse``
                              are accepted (they share ``BaseApplicationsSchema``).
            borough_name:     Display name of the borough / LPA.
            constraint_flags: Active planning constraint flags.  Penalty
                              modifiers are applied based on these flags.
            council_id:       IBex integer council ID, embedded in the stats
                              for downstream traceability.

        Returns:
            :class:`ScoringResult` with stats, base score, per-type rates,
            and the list of penalties that were applied.
        """
        if not applications:
            if not allow_mock:
                raise ValueError(
                    f"No IBex data for '{borough_name}' and mock mode is disabled. "
                    "Check that the IBex query returned results."
                )
            return self._mock_score(borough_name, constraint_flags, council_id)

        # --- Decision counts ------------------------------------------------
        approved  = sum(1 for a in applications if a.normalised_decision == NormalisedDecision.Approved)
        refused   = sum(1 for a in applications if a.normalised_decision == NormalisedDecision.Refused)
        pending   = sum(1 for a in applications if a.normalised_decision == NormalisedDecision.Validated)
        withdrawn = sum(1 for a in applications if a.normalised_decision == NormalisedDecision.Withdrawn)
        total     = len(applications)
        decided   = approved + refused

        # --- Rates ----------------------------------------------------------
        raw_approval_rate = round(approved / decided * 100, 1) if decided else 0.0
        raw_refusal_rate  = round(100.0 - raw_approval_rate, 1) if decided else 0.0

        # --- Per-type approval rates ----------------------------------------
        approval_by_type = self._approval_by_type(applications)

        # --- Decision timing ------------------------------------------------
        avg_decision_weeks = self._avg_decision_weeks(applications)

        # --- Trend ----------------------------------------------------------
        trend, trend_detail = self._detect_trend(applications)

        # --- Data quality ---------------------------------------------------
        if decided >= FULL_QUALITY_THRESHOLD:
            data_quality: str = "full"
        elif decided > 0:
            data_quality = "partial"
        else:
            data_quality = "mock"

        # --- Constraint penalties -------------------------------------------
        base_score, applied_penalties = self._apply_penalties(
            raw_approval_rate, constraint_flags
        )

        stats = BoroughStats(
            name=borough_name,
            council_id=council_id,
            total_applications=total,
            approved=approved,
            refused=refused,
            pending=pending,
            withdrawn=withdrawn,
            approval_rate=raw_approval_rate,
            refusal_rate=raw_refusal_rate,
            avg_decision_weeks=avg_decision_weeks,
            trend=trend,
            trend_detail=trend_detail,
            data_quality=data_quality,  # type: ignore[arg-type]
        )

        return ScoringResult(
            stats=stats,
            base_approval_score=base_score,
            approval_by_type=approval_by_type,
            applied_penalties=applied_penalties,
        )

    # ------------------------------------------------------------------
    # Real-data computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _approval_by_type(
        applications: list[BaseApplicationsSchema],
    ) -> dict[str, float]:
        """Compute approval rate per ``normalised_application_type``.

        Only decided applications (Approved/Refused) contribute.  Types
        with no decided applications are omitted.  Result is sorted
        alphabetically by type label.
        """
        buckets: dict[str, list[NormalisedDecision]] = defaultdict(list)
        for app in applications:
            if app.normalised_decision not in _DECIDED:
                continue
            label: str = (
                app.normalised_application_type.value
                if app.normalised_application_type is not None
                else "unknown"
            )
            buckets[label].append(app.normalised_decision)

        return {
            label: round(
                sum(1 for d in decisions if d == NormalisedDecision.Approved)
                / len(decisions)
                * 100,
                1,
            )
            for label, decisions in sorted(buckets.items())
        }

    @staticmethod
    def _avg_decision_weeks(
        applications: list[BaseApplicationsSchema],
    ) -> Optional[float]:
        """Average turnaround from ``application_date`` → ``decided_date`` in weeks.

        Skips records where either date is absent.  Guards against bad data
        by ignoring deltas that are negative or exceed :data:`MAX_DECISION_DAYS`.
        """
        samples: list[float] = []
        for app in applications:
            if app.application_date is None or app.decided_date is None:
                continue
            delta_days = (app.decided_date - app.application_date).days
            if 0 < delta_days < MAX_DECISION_DAYS:
                samples.append(delta_days / 7.0)
        return round(sum(samples) / len(samples), 1) if samples else None

    @staticmethod
    def _detect_trend(
        applications: list[BaseApplicationsSchema],
    ) -> tuple[Optional[TrendDirection], Optional[str]]:
        """Compare approval rate in the last 2 years against the prior period.

        Uses ``decided_date`` as the reference date; falls back to
        ``application_date`` when ``decided_date`` is absent.

        Returns ``(None, None)`` if either window contains fewer than
        :data:`TREND_MIN_SAMPLES` decided applications.
        """
        cutoff = date.today() - timedelta(days=TREND_WINDOW_DAYS)

        recent_approved = recent_decided = 0
        prior_approved  = prior_decided  = 0

        for app in applications:
            if app.normalised_decision not in _DECIDED:
                continue
            ref_date: Optional[date] = app.decided_date or app.application_date
            if ref_date is None:
                continue
            is_approved = app.normalised_decision == NormalisedDecision.Approved

            if ref_date >= cutoff:
                recent_decided += 1
                if is_approved:
                    recent_approved += 1
            else:
                prior_decided += 1
                if is_approved:
                    prior_approved += 1

        if recent_decided < TREND_MIN_SAMPLES or prior_decided < TREND_MIN_SAMPLES:
            return None, None

        recent_rate = recent_approved / recent_decided * 100
        prior_rate  = prior_approved  / prior_decided  * 100
        diff        = recent_rate - prior_rate

        if diff > 5:
            direction = TrendDirection.improving
        elif diff < -5:
            direction = TrendDirection.declining
        else:
            direction = TrendDirection.stable

        detail = (
            f"{recent_rate:.0f}% approval in last 2 years "
            f"vs {prior_rate:.0f}% prior"
        )
        return direction, detail

    @staticmethod
    def _apply_penalties(
        raw_approval_rate: float,
        flags: ConstraintFlags,
    ) -> tuple[int, list[tuple[str, int]]]:
        """Subtract constraint penalties from *raw_approval_rate*.

        Iterates :data:`CONSTRAINT_PENALTIES` in a fixed order so the
        returned ``applied_penalties`` list is deterministic regardless of
        flag evaluation order.

        Args:
            raw_approval_rate: Borough baseline approval rate (0–100 float).
            flags:             Active constraint flags for this site/borough.

        Returns:
            ``(base_approval_score, applied_penalties)`` where
            ``base_approval_score`` is clamped to 0–100.
        """
        score   = raw_approval_rate
        applied: list[tuple[str, int]] = []

        for attr, (label, penalty) in CONSTRAINT_PENALTIES.items():
            if getattr(flags, attr, False):
                score -= penalty
                applied.append((label, penalty))

        return int(max(0, min(100, round(score)))), applied

    # ------------------------------------------------------------------
    # Constraint inference from application data
    # ------------------------------------------------------------------

    @classmethod
    def infer_constraints(
        cls,
        applications: list[BaseApplicationsSchema],
    ) -> ConstraintFlags:
        """Heuristically infer :class:`~analysis.models.ConstraintFlags` from
        application metadata when a GIS constraint layer is unavailable.

        Detection logic:

        - ``listed_building_consent`` application type →
          ``listed_building = True`` and ``conservation_area = True``
        - ``conservation_area`` application type → ``conservation_area = True``
        - ``tree_preservation_order`` application type →
          ``tree_preservation_order = True``
        - Proposal text keyword scan for conservation area, listed building,
          and Article 4 signals.

        Limitations:
            Flood zones and Green Belt cannot be reliably inferred from
            application text alone — those flags remain ``False`` and must
            be set from a GIS layer.
        """
        conservation_area = False
        listed_building   = False
        tpo               = False
        article_4         = False

        for app in applications:
            # --- Application type signals -----------------------------------
            if app.normalised_application_type in _CONSERVATION_AREA_TYPES:
                conservation_area = True
            if app.normalised_application_type == NormalisedApplicationType.listed_building_consent:
                listed_building = True
            if app.normalised_application_type == NormalisedApplicationType.tree_preservation_order:
                tpo = True

            # --- Proposal text keyword scan ---------------------------------
            text = (app.proposal or "").lower()
            if "conservation area" in text:
                conservation_area = True
            if "listed building" in text:
                listed_building = True
            if "article 4" in text or "article four" in text:
                article_4 = True

            # Early exit if all detectable flags are already set
            if conservation_area and listed_building and tpo and article_4:
                break

        return ConstraintFlags(
            conservation_area=conservation_area,
            listed_building=listed_building,
            tree_preservation_order=tpo,
            article_4=article_4,
            data_source="heuristic",
        )

    # ------------------------------------------------------------------
    # Mock mode
    # ------------------------------------------------------------------

    def _mock_score(
        self,
        borough_name: str,
        constraint_flags: ConstraintFlags,
        council_id: Optional[int] = None,
    ) -> ScoringResult:
        """Generate deterministic, plausible stats when no real data is available.

        The RNG is seeded on *borough_name* so the same borough always
        produces the same mock numbers across calls.
        """
        rng = random.Random(hash(borough_name))

        total            = rng.randint(40, 200)
        raw_approval_rate = round(rng.uniform(55, 90), 1)
        raw_refusal_rate  = round(100.0 - raw_approval_rate, 1)

        approved  = round(total * raw_approval_rate / 100)
        refused   = max(0, total - approved - rng.randint(0, max(1, total // 10)))
        pending   = max(0, total - approved - refused - rng.randint(0, 5))
        withdrawn = max(0, total - approved - refused - pending)

        avg_decision_weeks = round(rng.uniform(8, 20), 1)

        # --- Mock trend -------------------------------------------------------
        recent_rate = raw_approval_rate + rng.uniform(-8, 8)
        prior_rate  = raw_approval_rate + rng.uniform(-8, 8)
        diff        = recent_rate - prior_rate
        if diff > 5:
            trend = TrendDirection.improving
        elif diff < -5:
            trend = TrendDirection.declining
        else:
            trend = TrendDirection.stable
        trend_detail = (
            f"{recent_rate:.0f}% approval in last 2 years "
            f"vs {prior_rate:.0f}% prior"
        )

        # --- Mock per-type approval rates ------------------------------------
        candidate_types = [
            "full planning application",
            "householder planning application",
            "lawful development",
            "listed building consent",
            "change of use",
        ]
        num_types = rng.randint(2, min(4, len(candidate_types)))
        mock_types = rng.sample(candidate_types, k=num_types)
        approval_by_type = {t: round(rng.uniform(50, 92), 1) for t in mock_types}

        # --- Constraint penalties --------------------------------------------
        base_score, applied_penalties = self._apply_penalties(
            raw_approval_rate, constraint_flags
        )

        stats = BoroughStats(
            name=borough_name,
            council_id=council_id,
            total_applications=total,
            approved=approved,
            refused=refused,
            pending=pending,
            withdrawn=withdrawn,
            approval_rate=raw_approval_rate,
            refusal_rate=raw_refusal_rate,
            avg_decision_weeks=avg_decision_weeks,
            trend=trend,
            trend_detail=trend_detail,
            data_quality="mock",
        )

        return ScoringResult(
            stats=stats,
            base_approval_score=base_score,
            approval_by_type=approval_by_type,
            applied_penalties=applied_penalties,
        )
