"""Data analyser — scores, ranks, and summarises adapter results.

Sits between the orchestrator output and the frontend response.  Takes
merged adapter results + a ``ParsedIntent`` and produces borough-level
statistics, site scores, and a plain-English summary.
"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta
from typing import Any, Optional

from pydantic import BaseModel, Field

from intent_parser.schema import ParsedIntent


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class BoroughStats(BaseModel):
    """Approval / refusal statistics for a single borough."""

    name: str
    total_applications: int = 0
    approved: int = 0
    refused: int = 0
    pending: int = 0
    withdrawn: int = 0
    approval_rate: float = Field(
        default=0.0,
        description="Approval rate as a percentage (0–100).",
    )
    refusal_rate: float = Field(
        default=0.0,
        description="Refusal rate as a percentage (0–100).",
    )
    avg_decision_weeks: Optional[float] = Field(
        default=None,
        description="Average time from submission to decision, in weeks.",
    )
    trend: Optional[str] = Field(
        default=None,
        description='"improving", "stable", or "declining" based on recent vs prior period.',
    )
    trend_detail: Optional[str] = Field(
        default=None,
        description="E.g. '72% approval in last 2 years vs 65% prior'.",
    )


class ScoredSite(BaseModel):
    """A single result item with a composite site score."""

    score: int = Field(
        ge=0,
        le=100,
        description="Composite site suitability score (0–100).",
    )
    title: str
    description: str
    council: Optional[str] = None
    location: Optional[dict] = None
    decision: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    penalties: list[str] = Field(
        default_factory=list,
        description="Human-readable list of score deductions applied.",
    )
    raw: dict = Field(
        default_factory=dict,
        description="Original adapter result.",
    )


class AnalysisResult(BaseModel):
    """Full analysis output — returned to the frontend."""

    summary: str = Field(
        description="Plain-English paragraph for the top of the results page.",
    )
    borough_stats: list[BoroughStats] = Field(default_factory=list)
    scored_sites: list[ScoredSite] = Field(
        default_factory=list,
        description="All results ranked by score (highest first).",
    )
    constraint_warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about constraints that matched (e.g. 'Site is in a flood zone').",
    )
    data_quality: str = Field(
        default="mock",
        description='"mock", "partial", or "full" — tells frontend how much to trust the numbers.',
    )


# ---------------------------------------------------------------------------
# Penalty weights for site scoring
# ---------------------------------------------------------------------------

_CONSTRAINT_PENALTIES: dict[str, int] = {
    "flood_risk": 25,
    "flood_zone": 25,
    "conservation_area": 15,
    "green_belt": 30,
    "article_4": 20,
    "listed_building": 10,
}


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


class DataAnalyser:
    """Analyses merged adapter results against a ParsedIntent.

    Works in two modes:

    - **Real mode** — when adapter results contain actual planning application
      data with ``decision``, ``date``, ``council`` fields.
    - **Mock mode** — when adapters return empty results (skeleton adapters),
      generates plausible mock statistics so the frontend can render.
    """

    def analyse(
        self,
        merged_results: list[dict[str, Any]],
        results_by_source: dict[str, list[dict[str, Any]]],
        intent: ParsedIntent,
    ) -> AnalysisResult:
        """Run the full analysis pipeline.

        Args:
            merged_results: Combined results from all adapters (may be empty).
            results_by_source: Results keyed by adapter name.
            intent: The original parsed intent.

        Returns:
            An ``AnalysisResult`` with stats, scored sites, and summary.
        """
        has_real_data = any(
            r.get("decision") is not None
            for r in merged_results
            if "error" not in r
        )

        if has_real_data:
            borough_stats = self._compute_borough_stats(merged_results)
            scored_sites = self._score_sites(merged_results, intent)
            constraint_warnings = self._check_constraints(merged_results, intent)
            data_quality = "full" if len(merged_results) >= 10 else "partial"
        else:
            borough_stats = self._mock_borough_stats(intent)
            scored_sites = self._mock_scored_sites(intent)
            constraint_warnings = self._mock_constraint_warnings(intent)
            data_quality = "mock"

        summary = self._build_summary(borough_stats, scored_sites, intent, data_quality)

        return AnalysisResult(
            summary=summary,
            borough_stats=borough_stats,
            scored_sites=scored_sites,
            constraint_warnings=constraint_warnings,
            data_quality=data_quality,
        )

    # ------------------------------------------------------------------
    # Real data analysis
    # ------------------------------------------------------------------

    def _compute_borough_stats(
        self,
        results: list[dict[str, Any]],
    ) -> list[BoroughStats]:
        """Aggregate approval/refusal stats per borough from real results."""
        by_borough: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            council = r.get("council", "Unknown")
            by_borough.setdefault(council, []).append(r)

        stats: list[BoroughStats] = []
        for name, items in sorted(by_borough.items()):
            approved = sum(1 for i in items if _decision_is(i, "approved"))
            refused = sum(1 for i in items if _decision_is(i, "refused"))
            pending = sum(1 for i in items if _decision_is(i, "pending"))
            withdrawn = sum(1 for i in items if _decision_is(i, "withdrawn"))
            total = len(items)
            decided = approved + refused

            avg_weeks = self._avg_decision_weeks(items)
            trend, trend_detail = self._detect_trend(items)

            stats.append(BoroughStats(
                name=name,
                total_applications=total,
                approved=approved,
                refused=refused,
                pending=pending,
                withdrawn=withdrawn,
                approval_rate=round(approved / decided * 100, 1) if decided else 0.0,
                refusal_rate=round(refused / decided * 100, 1) if decided else 0.0,
                avg_decision_weeks=avg_weeks,
                trend=trend,
                trend_detail=trend_detail,
            ))

        stats.sort(key=lambda s: s.approval_rate, reverse=True)
        return stats

    def _avg_decision_weeks(self, items: list[dict[str, Any]]) -> Optional[float]:
        """Calculate average decision time in weeks from submission to decision date."""
        weeks: list[float] = []
        for item in items:
            raw = item.get("raw", {})
            submitted = raw.get("date_submitted") or raw.get("date_received")
            decided = raw.get("date_decided") or raw.get("date_decision")
            if submitted and decided:
                try:
                    d_sub = datetime.fromisoformat(submitted).date()
                    d_dec = datetime.fromisoformat(decided).date()
                    delta = (d_dec - d_sub).days
                    if 0 < delta < 365 * 3:
                        weeks.append(delta / 7)
                except (ValueError, TypeError):
                    continue
        return round(sum(weeks) / len(weeks), 1) if weeks else None

    def _detect_trend(
        self,
        items: list[dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        """Compare approval rate in last 2 years vs the prior period."""
        today = date.today()
        cutoff = today - timedelta(days=2 * 365)

        recent_approved = 0
        recent_decided = 0
        prior_approved = 0
        prior_decided = 0

        for item in items:
            d = _parse_date(item.get("date"))
            if d is None:
                continue
            is_approved = _decision_is(item, "approved")
            is_decided = _decision_is(item, "approved") or _decision_is(item, "refused")
            if not is_decided:
                continue
            if d >= cutoff:
                recent_decided += 1
                if is_approved:
                    recent_approved += 1
            else:
                prior_decided += 1
                if is_approved:
                    prior_approved += 1

        if recent_decided < 3 or prior_decided < 3:
            return None, None

        recent_rate = recent_approved / recent_decided * 100
        prior_rate = prior_approved / prior_decided * 100
        diff = recent_rate - prior_rate

        if diff > 5:
            trend = "improving"
        elif diff < -5:
            trend = "declining"
        else:
            trend = "stable"

        detail = f"{recent_rate:.0f}% approval in last 2 years vs {prior_rate:.0f}% prior"
        return trend, detail

    def _score_sites(
        self,
        results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[ScoredSite]:
        """Score each result 0–100 based on relevance and constraint penalties."""
        # Build set of constraints the user wants to avoid.
        avoid_categories = {
            c.category for c in intent.constraints if c.type == "avoid"
        }

        scored: list[ScoredSite] = []
        for r in results:
            if "error" in r:
                continue

            base_score = int(r.get("relevance_score", 0.5) * 100)
            penalties: list[str] = []

            # Penalty: decision was refused.
            if _decision_is(r, "refused"):
                base_score = max(0, base_score - 15)
                penalties.append("-15 refused application")

            # Penalty: matches an avoided constraint.
            raw = r.get("raw", {})
            for category, weight in _CONSTRAINT_PENALTIES.items():
                if category in avoid_categories:
                    # Check if the raw result flags this constraint.
                    in_zone = (
                        raw.get(category)
                        or raw.get(f"in_{category}")
                        or raw.get("designation") == category
                    )
                    if in_zone:
                        base_score = max(0, base_score - weight)
                        penalties.append(f"-{weight} {category.replace('_', ' ')}")

            # Bonus: approved and matches development type.
            if _decision_is(r, "approved"):
                base_score = min(100, base_score + 10)

            scored.append(ScoredSite(
                score=max(0, min(100, base_score)),
                title=r.get("title", "Untitled"),
                description=r.get("description", ""),
                council=r.get("council"),
                location=r.get("location"),
                decision=r.get("decision"),
                date=r.get("date"),
                source=r.get("source"),
                penalties=penalties,
                raw=r,
            ))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def _check_constraints(
        self,
        results: list[dict[str, Any]],
        intent: ParsedIntent,
    ) -> list[str]:
        """Generate warnings for any constraint violations found in results."""
        avoid_categories = {
            c.category: c for c in intent.constraints if c.type == "avoid"
        }
        warnings: list[str] = []
        seen: set[str] = set()

        for r in results:
            raw = r.get("raw", {})
            for category in avoid_categories:
                if category in seen:
                    continue
                in_zone = (
                    raw.get(category)
                    or raw.get(f"in_{category}")
                    or raw.get("designation") == category
                )
                if in_zone:
                    label = category.replace("_", " ").title()
                    council = r.get("council", "the area")
                    warnings.append(
                        f"Warning: {council} has sites in a {label} zone. "
                        f"You asked to avoid this."
                    )
                    seen.add(category)

        return warnings

    # ------------------------------------------------------------------
    # Mock data generation
    # ------------------------------------------------------------------

    def _mock_borough_stats(self, intent: ParsedIntent) -> list[BoroughStats]:
        """Generate plausible mock stats for the resolved councils."""
        councils = intent.location.resolved_councils
        if not councils:
            councils = ["Hackney", "Lambeth", "Southwark"]

        rng = random.Random(hash(intent.raw_query))
        stats: list[BoroughStats] = []

        for name in councils[:8]:
            total = rng.randint(40, 200)
            approval_rate = rng.uniform(55, 90)
            approved = round(total * approval_rate / 100)
            refused = total - approved - rng.randint(0, max(1, total // 10))
            refused = max(0, refused)
            pending = max(0, total - approved - refused - rng.randint(0, 5))
            withdrawn = total - approved - refused - pending

            recent_rate = approval_rate + rng.uniform(-8, 8)
            prior_rate = approval_rate + rng.uniform(-8, 8)
            diff = recent_rate - prior_rate
            trend = "improving" if diff > 5 else ("declining" if diff < -5 else "stable")
            trend_detail = f"{recent_rate:.0f}% approval in last 2 years vs {prior_rate:.0f}% prior"

            stats.append(BoroughStats(
                name=name,
                total_applications=total,
                approved=approved,
                refused=refused,
                pending=pending,
                withdrawn=withdrawn,
                approval_rate=round(approval_rate, 1),
                refusal_rate=round(100 - approval_rate, 1),
                avg_decision_weeks=round(rng.uniform(8, 20), 1),
                trend=trend,
                trend_detail=trend_detail,
            ))

        stats.sort(key=lambda s: s.approval_rate, reverse=True)
        return stats

    def _mock_scored_sites(self, intent: ParsedIntent) -> list[ScoredSite]:
        """Generate plausible mock scored sites."""
        councils = intent.location.resolved_councils or ["Hackney"]
        dev_desc = intent.development.description
        category = intent.development.category
        rng = random.Random(hash(intent.raw_query) + 1)

        avoid_categories = {c.category for c in intent.constraints if c.type == "avoid"}

        mock_decisions = ["Approved", "Approved", "Approved", "Refused", "Approved", "Pending"]
        mock_app_types = {
            "residential": ["Full planning application — new dwelling", "Outline planning — residential"],
            "change_of_use": ["Change of use — Class E to C3", "Prior approval — office to residential"],
            "home_improvement": ["Householder — rear extension", "Householder — loft conversion"],
            "commercial": ["Full planning application — commercial", "Change of use — retail"],
        }
        titles = mock_app_types.get(category, [f"Full planning application — {dev_desc}"])

        sites: list[ScoredSite] = []
        for i in range(rng.randint(5, 12)):
            council = rng.choice(councils)
            decision = rng.choice(mock_decisions)
            base_score = rng.randint(50, 95)
            penalties: list[str] = []

            if decision == "Refused":
                base_score = max(0, base_score - 15)
                penalties.append("-15 refused application")

            if "flood_risk" in avoid_categories and rng.random() < 0.2:
                base_score = max(0, base_score - 25)
                penalties.append("-25 flood risk")
            if "conservation_area" in avoid_categories and rng.random() < 0.15:
                base_score = max(0, base_score - 15)
                penalties.append("-15 conservation area")
            if "green_belt" in avoid_categories and rng.random() < 0.1:
                base_score = max(0, base_score - 30)
                penalties.append("-30 green belt")

            days_ago = rng.randint(30, 3 * 365)
            app_date = (date.today() - timedelta(days=days_ago)).isoformat()

            sites.append(ScoredSite(
                score=max(0, min(100, base_score)),
                title=rng.choice(titles),
                description=f"Application in {council} — {decision.lower()}",
                council=council,
                location=None,
                decision=decision,
                date=app_date,
                source="mock",
                penalties=penalties,
                raw={"_mock": True},
            ))

        sites.sort(key=lambda s: s.score, reverse=True)
        return sites

    def _mock_constraint_warnings(self, intent: ParsedIntent) -> list[str]:
        """Generate warnings for mock mode based on the intent's constraints."""
        warnings: list[str] = []
        councils = intent.location.resolved_councils or ["the area"]
        for c in intent.constraints:
            if c.type == "avoid":
                label = c.category.replace("_", " ").title()
                council = councils[0] if councils else "the area"
                warnings.append(
                    f"Warning: Some sites in {council} may be in a {label} zone. "
                    f"Real data needed to confirm."
                )
        return warnings

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        borough_stats: list[BoroughStats],
        scored_sites: list[ScoredSite],
        intent: ParsedIntent,
        data_quality: str,
    ) -> str:
        """Build a plain-English summary paragraph for the results page."""
        parts: list[str] = []

        dev = intent.development.description
        location = intent.location.raw_text or "your target area"

        if data_quality == "mock":
            parts.append(
                f"Based on simulated data for {dev} in {location} "
                f"(real API data not yet connected):"
            )
        else:
            parts.append(f"Analysis of planning applications for {dev} in {location}:")

        # Borough highlights.
        if borough_stats:
            best = borough_stats[0]
            parts.append(
                f" {best.name} has the highest approval rate at {best.approval_rate:.0f}%"
                f" ({best.approved} approved out of {best.total_applications} applications)."
            )
            if best.avg_decision_weeks is not None:
                parts.append(
                    f" Average decision time is {best.avg_decision_weeks:.0f} weeks."
                )
            if best.trend and best.trend != "stable":
                direction = "up" if best.trend == "improving" else "down"
                parts.append(f" Trend is {direction} — {best.trend_detail}.")

            if len(borough_stats) > 1:
                worst = borough_stats[-1]
                parts.append(
                    f" {worst.name} has the lowest at {worst.approval_rate:.0f}%."
                )

        # Constraint warnings summary.
        avoid_count = sum(1 for c in intent.constraints if c.type == "avoid")
        if avoid_count:
            flagged = sum(
                1 for s in scored_sites if s.penalties
            )
            parts.append(
                f" {flagged} of {len(scored_sites)} sites flagged"
                f" for constraint violations."
            )

        # Top site.
        if scored_sites:
            top = scored_sites[0]
            parts.append(f" Top-scoring site: {top.title} (score {top.score}/100).")

        return "".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decision_is(result: dict[str, Any], decision: str) -> bool:
    """Check if a result's decision matches (case-insensitive)."""
    d = result.get("decision")
    return d is not None and d.lower() == decision.lower()


def _parse_date(value: Any) -> Optional[date]:
    """Try to parse a date string into a date object."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (ValueError, TypeError):
        return None
