"""Comparable application finder — selects the most relevant precedent
applications from a pool of IBex results for a given user intent.

Powers the "we based this on these real precedents" section of the
due diligence report.  Each comparable is returned as a
:class:`~analysis.models.ComparableApplication` with a numeric
``similarity_score`` (0–1) and a list of plain-English
``similarity_reasons`` explaining why it was selected.

Similarity dimensions (and default weights):

=================  ======  ================================================
Dimension          Weight  Logic
=================  ======  ================================================
Application type    0.30   Does the normalised_application_type match what
                           the intent's development category expects?
Borough             0.25   Is the application in one of the intent's
                           resolved councils?
Unit count          0.20   How close is the application's unit count to the
                           intent's requested unit_count?
Project type        0.10   Does the IBex project_type match the expected
                           category (e.g. small_residential)?
Recency             0.10   Newer decisions are slightly preferred (linear
                           decay over 5 years).
Keyword overlap     0.05   Do any of the intent's keywords appear in the
                           proposal text?
=================  ======  ================================================

Only *decided* applications (Approved / Refused) are considered — pending,
withdrawn, and unknown outcomes are excluded.

Usage::

    from analysis.comparables import ComparableFinder

    finder = ComparableFinder()
    comps  = finder.find(parsed_intent, ibex_applications, top_n=5)
    for c in comps:
        print(c.planning_reference, c.similarity_score, c.similarity_reasons)
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from hashbrowns.ibex.models import (
    BaseApplicationsSchema,
    NormalisedApplicationType,
    NormalisedDecision,
    ProjectType,
)

from intent_parser.schema import ParsedIntent

from .models import ComparableApplication


# ---------------------------------------------------------------------------
# Similarity weights — must sum to 1.0
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "application_type": 0.30,
    "borough":          0.25,
    "unit_count":       0.20,
    "project_type":     0.10,
    "recency":          0.10,
    "keyword":          0.05,
}

# ---------------------------------------------------------------------------
# Intent category → expected IBex values
# ---------------------------------------------------------------------------

#: Maps intent ``development.category`` to the normalised application types
#: that would represent a similar application in the IBex data.
CATEGORY_TO_APP_TYPES: dict[str, frozenset[NormalisedApplicationType]] = {
    "residential": frozenset({
        NormalisedApplicationType.full_planning_application,
    }),
    "commercial": frozenset({
        NormalisedApplicationType.full_planning_application,
    }),
    "mixed_use": frozenset({
        NormalisedApplicationType.full_planning_application,
    }),
    "home_improvement": frozenset({
        NormalisedApplicationType.householder_planning_application,
    }),
    "change_of_use": frozenset({
        NormalisedApplicationType.change_of_use,
    }),
    "infrastructure": frozenset({
        NormalisedApplicationType.full_planning_application,
    }),
    "hospitality": frozenset({
        NormalisedApplicationType.full_planning_application,
        NormalisedApplicationType.change_of_use,
    }),
    "industrial": frozenset({
        NormalisedApplicationType.full_planning_application,
    }),
    "retail": frozenset({
        NormalisedApplicationType.full_planning_application,
        NormalisedApplicationType.change_of_use,
    }),
}

#: Maps intent ``development.category`` to expected IBex ``ProjectType`` values.
CATEGORY_TO_PROJECT_TYPES: dict[str, frozenset[ProjectType]] = {
    "residential": frozenset({
        ProjectType.small_residential,
        ProjectType.medium_residential,
        ProjectType.large_residential,
    }),
    "home_improvement": frozenset({ProjectType.home_improvement}),
    "mixed_use": frozenset({ProjectType.mixed}),
}

#: Linear recency decay window in days.  An application decided today
#: scores 1.0; one decided ``RECENCY_HORIZON_DAYS`` ago scores 0.0.
RECENCY_HORIZON_DAYS: int = 5 * 365

#: Only decided applications are useful as precedents.
_DECIDED: frozenset[NormalisedDecision] = frozenset(
    {NormalisedDecision.Approved, NormalisedDecision.Refused}
)

__all__ = ["ComparableFinder", "WEIGHTS"]


# ---------------------------------------------------------------------------
# Finder
# ---------------------------------------------------------------------------


class ComparableFinder:
    """Selects the most relevant precedent applications for a given intent.

    Stateless — safe to instantiate once and call :meth:`find` for
    different intents / application pools.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find(
        self,
        intent: ParsedIntent,
        applications: list[BaseApplicationsSchema],
        top_n: int = 5,
    ) -> list[ComparableApplication]:
        """Find the *top_n* most similar decided applications.

        Args:
            intent:       The user's parsed development intent.
            applications: Pool of IBex application records to search.
                          Both ``SearchResponse`` and ``ApplicationsResponse``
                          are accepted.
            top_n:        Maximum number of comparables to return.

        Returns:
            Up to *top_n* :class:`ComparableApplication` objects sorted
            by ``similarity_score`` descending.  Empty list if no decided
            applications exist in the pool.
        """
        scored: list[tuple[float, list[str], BaseApplicationsSchema]] = []

        for app in applications:
            if app.normalised_decision not in _DECIDED:
                continue
            score, reasons = self._score_application(app, intent)
            scored.append((score, reasons, app))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            self._to_comparable(app, score, reasons)
            for score, reasons, app in scored[:top_n]
        ]

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------

    def _score_application(
        self,
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, list[str]]:
        """Compute a weighted 0–1 similarity score and collect reasons.

        Each dimension method returns ``(sub_score, reason_or_None)``.
        Sub-scores are combined via :data:`WEIGHTS`.  Reasons are
        collected for any dimension that contributes positively.
        """
        total = 0.0
        reasons: list[str] = []

        dimensions: list[tuple[str, float, Optional[str]]] = [
            ("application_type", *self._app_type_score(app, intent)),
            ("borough",          *self._borough_score(app, intent)),
            ("unit_count",       *self._unit_count_score(app, intent)),
            ("project_type",     *self._project_type_score(app, intent)),
            ("recency",          *self._recency_score(app)),
            ("keyword",          *self._keyword_score(app, intent)),
        ]

        for name, sub_score, reason in dimensions:
            total += WEIGHTS[name] * sub_score
            if reason is not None:
                reasons.append(reason)

        # Add decision context as a reason
        if app.normalised_decision == NormalisedDecision.Approved:
            reasons.append("Approved — supports feasibility")
        else:
            reasons.append("Refused — useful refusal precedent")

        return round(total, 4), reasons

    # ------------------------------------------------------------------
    # Dimension scorers — each returns (0–1 score, optional reason)
    # ------------------------------------------------------------------

    @staticmethod
    def _app_type_score(
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, Optional[str]]:
        """Score how well the application type matches the intent category."""
        expected = CATEGORY_TO_APP_TYPES.get(intent.development.category)
        if expected is None:
            # Unknown category — partial credit if app has any type set
            return (0.3, None) if app.normalised_application_type else (0.0, None)

        if app.normalised_application_type in expected:
            label = app.normalised_application_type.value
            return 1.0, f"Same application type ({label})"

        # Partial credit for related types (e.g. listed building consent
        # for a residential project in a conservation area)
        if app.normalised_application_type is not None:
            return 0.2, None

        return 0.0, None

    @staticmethod
    def _borough_score(
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, Optional[str]]:
        """Score whether the application is in one of the target boroughs."""
        councils = intent.location.resolved_councils
        if not councils:
            # No target boroughs resolved — give neutral credit
            return 0.5, None

        if app.council_name in councils:
            return 1.0, f"Same borough ({app.council_name})"

        # Different borough — some value as a nearby precedent
        return 0.1, None

    @staticmethod
    def _unit_count_score(
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, Optional[str]]:
        """Score proximity of the application's unit count to the intent."""
        target_units = intent.development.unit_count
        app_units = _extract_unit_count(app)

        if target_units is None and app_units is None:
            # Neither specifies units — neutral
            return 0.5, None

        if target_units is None or app_units is None:
            # Only one side has units — weak signal
            return 0.2, None

        # Both have unit counts — score based on proximity
        diff = abs(app_units - target_units)
        denominator = max(target_units, 1)
        ratio = min(diff / denominator, 1.0)
        score = 1.0 - ratio

        if score >= 0.7:
            return score, f"Similar scale ({app_units} units vs {target_units} requested)"
        if score >= 0.4:
            return score, f"Comparable scale ({app_units} units vs {target_units} requested)"
        return score, None

    @staticmethod
    def _project_type_score(
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, Optional[str]]:
        """Score whether the IBex project_type matches the intent category."""
        project_type: Optional[ProjectType] = getattr(app, "project_type", None)
        if project_type is None:
            return 0.3, None  # no data — partial credit

        expected = CATEGORY_TO_PROJECT_TYPES.get(intent.development.category)
        if expected is None:
            return 0.3, None  # unknown category — partial credit

        if project_type in expected:
            return 1.0, f"Same project type ({project_type.value})"

        return 0.1, None

    @staticmethod
    def _recency_score(
        app: BaseApplicationsSchema,
    ) -> tuple[float, Optional[str]]:
        """Score recency with linear decay over the horizon window."""
        ref_date = app.decided_date or app.application_date
        if ref_date is None:
            return 0.3, None  # no date — partial credit

        days_ago = (date.today() - ref_date).days
        if days_ago < 0:
            days_ago = 0  # future date (data quirk) — treat as today

        score = max(0.0, 1.0 - days_ago / RECENCY_HORIZON_DAYS)

        if score >= 0.6:
            year = ref_date.year
            return score, f"Recent decision ({year})"

        return score, None

    @staticmethod
    def _keyword_score(
        app: BaseApplicationsSchema,
        intent: ParsedIntent,
    ) -> tuple[float, Optional[str]]:
        """Score keyword overlap between intent keywords/tags and proposal text."""
        # Combine intent keywords and raw_tags into one search set
        search_terms: set[str] = set()
        for kw in intent.keywords:
            search_terms.add(kw.lower())
        for tag in intent.development.raw_tags:
            search_terms.add(tag.lower())

        if not search_terms:
            return 0.5, None  # no keywords to check — neutral

        proposal = (app.proposal or "").lower()
        if not proposal:
            return 0.0, None

        matched = [term for term in search_terms if term in proposal]
        score = len(matched) / len(search_terms)

        if matched:
            display = ", ".join(sorted(matched)[:3])
            suffix = f" (+{len(matched) - 3} more)" if len(matched) > 3 else ""
            return score, f"Keyword match: {display}{suffix}"

        return 0.0, None

    # ------------------------------------------------------------------
    # Conversion helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_comparable(
        app: BaseApplicationsSchema,
        score: float,
        reasons: list[str],
    ) -> ComparableApplication:
        """Convert a scored IBex application to a ComparableApplication."""
        decision_weeks: Optional[float] = None
        if app.application_date is not None and app.decided_date is not None:
            delta_days = (app.decided_date - app.application_date).days
            if 0 < delta_days < 3 * 365:
                decision_weeks = round(delta_days / 7.0, 1)

        return ComparableApplication(
            planning_reference=app.planning_reference,
            council_name=app.council_name,
            url=app.url,
            normalised_decision=app.normalised_decision,
            normalised_application_type=app.normalised_application_type,
            project_type=getattr(app, "project_type", None),
            proposal=app.proposal,
            raw_address=app.raw_address,
            application_date=app.application_date,
            decided_date=app.decided_date,
            decision_weeks=decision_weeks,
            similarity_score=min(score, 1.0),
            similarity_reasons=reasons,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_unit_count(app: BaseApplicationsSchema) -> Optional[int]:
    """Best-effort extraction of unit count from an IBex application.

    Checks, in order:
    1. ``num_new_houses`` (direct field on extended response types)
    2. ``proposed_unit_mix.total_proposed_residential_units``

    Returns ``None`` if neither is available.
    """
    # num_new_houses is on SearchResponse / ApplicationsResponse, not base
    num_houses: Optional[int] = getattr(app, "num_new_houses", None)
    if num_houses is not None:
        return num_houses

    unit_mix = getattr(app, "proposed_unit_mix", None)
    if unit_mix is not None:
        total = getattr(unit_mix, "total_proposed_residential_units", None)
        if total is not None:
            return total

    return None
