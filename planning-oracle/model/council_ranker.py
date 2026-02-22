"""Stage 1: Council shortlisting model.

Ranks councils by their expected likelihood of approving a given proposal
type, so the downstream pipeline can focus on the most relevant authorities.

Scoring formula (per council)::

    score = (w_approval  * approval_rate_for_project_type
           + w_speed     * normalised_decision_speed
           + w_activity  * normalised_activity_level
           + w_volume    * normalised_homes_volume)

All components are normalised to [0, 1].  Weights default to:
approval=0.45, speed=0.20, activity=0.20, volume=0.15.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from config.settings import Settings, get_settings
from data.schema import CouncilStats
from inference.parser import ProposalIntent

logger = logging.getLogger(__name__)

# Default scoring weights.
_W_APPROVAL: float = 0.45
_W_SPEED: float = 0.20
_W_ACTIVITY: float = 0.20
_W_VOLUME: float = 0.15

_ACTIVITY_SCORES: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
}

# Map ProposalIntent.project_type to CouncilStats.average_decision_time keys.
_PROJECT_TYPE_MAP: dict[str, str] = {
    "small residential": "residential",
    "medium residential": "residential",
    "large residential": "residential",
    "home improvement": "residential",
    "mixed": "commercial",
}


class CouncilRanker:
    """Rank councils by predicted approval affinity for a proposal.

    Parameters:
        w_approval: Weight for approval rate component.
        w_speed: Weight for decision speed component.
        w_activity: Weight for development activity component.
        w_volume: Weight for new-homes volume component.
        settings: Application settings.
    """

    def __init__(
        self,
        *,
        w_approval: float = _W_APPROVAL,
        w_speed: float = _W_SPEED,
        w_activity: float = _W_ACTIVITY,
        w_volume: float = _W_VOLUME,
        settings: Optional[Settings] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._w_approval = w_approval
        self._w_speed = w_speed
        self._w_activity = w_activity
        self._w_volume = w_volume

    def rank_councils(
        self,
        intent: ProposalIntent,
        council_stats: dict[int, CouncilStats],
        top_k: int = 15,
    ) -> list[tuple[int, float]]:
        """Score and rank councils for a parsed proposal.

        Args:
            intent: Structured proposal intent from the parser.
            council_stats: Mapping of ``council_id`` to
                :class:`CouncilStats` instances.
            top_k: Number of top councils to return.

        Returns:
            List of ``(council_id, score)`` tuples sorted descending by
            score.  Scores are in ``[0, 1]``.
        """
        if not council_stats:
            return []

        project_key = _PROJECT_TYPE_MAP.get(intent.project_type, "residential")

        # ── collect raw values for normalisation ─────────────────────
        raw_scores: list[tuple[int, float, float, float, float]] = []
        all_speeds: list[float] = []
        all_volumes: list[float] = []

        for cid, stats in council_stats.items():
            # API returns approval_rate as 0-100 percentage; normalise to 0-1.
            approval = (stats.approval_rate or 0.0) / 100.0

            # Decision speed for the relevant project type
            speed = 0.0
            if stats.average_decision_time:
                speed = stats.average_decision_time.get(project_key, 0.0)
            all_speeds.append(speed)

            # Activity level
            activity_str = (
                stats.council_development_activity_level or ""
            ).lower()
            activity = _ACTIVITY_SCORES.get(activity_str, 0.0)

            # New homes volume
            volume = float(stats.number_of_new_homes_approved or 0)
            all_volumes.append(volume)

            raw_scores.append((cid, approval, speed, activity, volume))

        # ── normalise speed and volume ───────────────────────────────
        max_speed = max(all_speeds) if all_speeds else 1.0
        max_volume = max(all_volumes) if all_volumes else 1.0

        scored: list[tuple[int, float]] = []
        for cid, approval, speed, activity, volume in raw_scores:
            # Faster is better → invert so lower days = higher score
            norm_speed = 1.0 - (speed / max_speed) if max_speed > 0 else 0.5
            norm_volume = volume / max_volume if max_volume > 0 else 0.0

            score = (
                self._w_approval * approval
                + self._w_speed * norm_speed
                + self._w_activity * activity
                + self._w_volume * norm_volume
            )

            scored.append((cid, round(score, 6)))

        # ── sort and return top-k ────────────────────────────────────
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        logger.info(
            "Ranked %d councils for '%s' (project_key=%s), top score=%.4f",
            len(scored), intent.project_type, project_key,
            top[0][1] if top else 0.0,
        )
        return top

    # ── persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialise the ranker configuration to disk."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "w_approval": self._w_approval,
                    "w_speed": self._w_speed,
                    "w_activity": self._w_activity,
                    "w_volume": self._w_volume,
                },
                f,
            )
        logger.info("CouncilRanker saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CouncilRanker:
        """Load a previously saved ranker."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls(
            w_approval=state["w_approval"],
            w_speed=state["w_speed"],
            w_activity=state["w_activity"],
            w_volume=state["w_volume"],
        )
        logger.info("CouncilRanker loaded from %s", path)
        return inst
