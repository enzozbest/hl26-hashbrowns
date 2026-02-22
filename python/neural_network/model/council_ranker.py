"""Stage 1: Council ranking via per-council model predictions.

Ranks councils by running the full :class:`ApprovalModel` with each
council's specific feature vector, producing *personalised* approval
probabilities conditioned on the proposal text, application features,
and council context.

The model (~80–90k parameters) is small enough that a single batched
forward pass over all councils (~300–400 in England) is trivially
cheap — a few milliseconds on CPU.  This eliminates the need for a
heuristic pre-filter and ensures the ranking is fully proposal-aware.

A lightweight heuristic fallback (:meth:`rank_councils`) is retained
for use cases where the model is unavailable (e.g. cold-start before
training).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Protocol

import torch

from neural_network.config.settings import Settings, get_settings
from neural_network.data.schema import CouncilStats
from neural_network.inference.parser import ProposalIntent
from neural_network.model.approval_model import ApprovalModel
from neural_network.model.calibration import TemperatureScaler

logger = logging.getLogger(__name__)

# Default scoring weights (heuristic fallback only).
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


# ── Council feature builder protocol ─────────────────────────────────────────


class CouncilFeatureBuilder(Protocol):
    """Protocol for callables that build a council feature tensor.

    Implementations should return a 1-D tensor of shape
    ``(num_council_features,)`` matching the council branch input
    dimensionality of the :class:`ApprovalModel`.
    """

    def __call__(
        self,
        stats: CouncilStats,
        intent: ProposalIntent,
    ) -> torch.Tensor: ...


# ── Ranker ───────────────────────────────────────────────────────────────────


class CouncilRanker:
    """Rank councils by model-predicted approval probability.

    Primary method: :meth:`rank_councils_with_model` — runs the neural
    network on every council in a single batched forward pass.

    Fallback: :meth:`rank_councils` — lightweight heuristic for when
    the model is unavailable.

    Parameters:
        w_approval: Weight for approval rate component (heuristic only).
        w_speed: Weight for decision speed component (heuristic only).
        w_activity: Weight for activity component (heuristic only).
        w_volume: Weight for homes volume component (heuristic only).
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

    # ── Model-based ranking (primary) ────────────────────────────────

    @torch.no_grad()
    def rank_councils_with_model(
        self,
        intent: ProposalIntent,
        council_stats: dict[int, CouncilStats],
        *,
        model: ApprovalModel,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_feature_builder: CouncilFeatureBuilder,
        scaler: Optional[TemperatureScaler] = None,
        top_k: int = 15,
        device: Optional[torch.device] = None,
    ) -> list[dict[str, Any]]:
        """Rank all councils by running the model in a single batched pass.

        For each council, the model receives the *same* text embedding
        and application features but a *different* council feature
        vector, producing a personalised approval probability.

        With ~80–90k model parameters and ~300–400 councils, the full
        batch completes in a few milliseconds on CPU.

        Args:
            intent: Structured proposal intent from the parser.
            council_stats: Mapping of ``council_id`` to
                :class:`CouncilStats` instances.
            model: Trained :class:`ApprovalModel` (set to eval
                internally).
            text_embedding: Pre-computed text embedding for the
                proposal, shape ``(1, text_embed_dim)`` or
                ``(text_embed_dim,)``.
            app_features: Application feature vector, shape
                ``(1, num_app_features)`` or ``(num_app_features,)``.
            council_feature_builder: Builds a council feature tensor
                from :class:`CouncilStats` and :class:`ProposalIntent`.
                Must return shape ``(num_council_features,)``.
            scaler: Fitted :class:`TemperatureScaler` for calibrated
                probabilities (*None* → raw sigmoid).
            top_k: Number of top councils to return.
            device: Inference device (inferred from model if *None*).

        Returns:
            List of dicts sorted descending by
            ``approval_probability``::

                [
                    {
                        "council_id": int,
                        "approval_probability": float,
                    },
                    ...
                ]
        """
        if not council_stats:
            return []

        if device is None:
            device = next(model.parameters()).device

        model.eval()

        # ── Ensure shared inputs have batch dim ──────────────────────
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        if app_features.dim() == 1:
            app_features = app_features.unsqueeze(0)

        # ── Build council feature batch ──────────────────────────────
        council_ids: list[int] = []
        council_tensors: list[torch.Tensor] = []

        for cid, stats in council_stats.items():
            feat = council_feature_builder(stats, intent)

            if not isinstance(feat, torch.Tensor):
                feat = torch.as_tensor(feat, dtype=torch.float32)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)

            council_tensors.append(feat)
            council_ids.append(cid)

        n = len(council_tensors)
        if n == 0:
            return []

        # Stack: (N, num_council_features)
        council_batch = torch.cat(council_tensors, dim=0).to(device)

        # Expand shared inputs: (1, D) → (N, D)
        text_batch = text_embedding.to(device).expand(n, -1)
        app_batch = app_features.to(device).expand(n, -1)

        # ── Single batched forward pass ──────────────────────────────
        logits = model(text_batch, app_batch, council_batch).squeeze(-1)

        # ── Calibrate ────────────────────────────────────────────────
        if scaler is not None:
            probs = scaler.calibrate(logits)
        else:
            probs = torch.sigmoid(logits)

        probs_list = probs.cpu().tolist()

        # Handle scalar case (single council)
        if not isinstance(probs_list, list):
            probs_list = [probs_list]

        # ── Assemble & sort ──────────────────────────────────────────
        results = [
            {
                "council_id": cid,
                "council_name": (
                    council_stats[cid].council_name
                    if cid in council_stats and council_stats[cid].council_name
                    else f"Council {cid}"
                ),
                "approval_probability": round(p, 4),
            }
            for cid, p in zip(council_ids, probs_list)
        ]

        results.sort(
            key=lambda r: r["approval_probability"], reverse=True,
        )
        results = results[:top_k]

        logger.info(
            "Ranked %d/%d councils for '%s': "
            "top=%.4f (council %d), bottom=%.4f (council %d)",
            len(results),
            n,
            intent.project_type,
            results[0]["approval_probability"],
            results[0]["council_id"],
            results[-1]["approval_probability"],
            results[-1]["council_id"],
        )

        return results

    # ── Heuristic fallback ───────────────────────────────────────────

    def rank_councils(
        self,
        intent: ProposalIntent,
        council_stats: dict[int, CouncilStats],
        top_k: int = 15,
        region: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Lightweight heuristic ranking (no model required).

        Retained as a fallback for cold-start or when the model is
        unavailable.  Does **not** consider the proposal text or
        application-level features.

        When *region* is provided, only councils whose ``region`` field
        matches are considered.  Normalisation (speed, volume) is then
        computed within-region so scores reflect relative standing among
        regional peers.

        Args:
            intent: Structured proposal intent from the parser.
            council_stats: Mapping of ``council_id`` to
                :class:`CouncilStats` instances.
            top_k: Number of top councils to return.
            region: If set, restrict ranking to councils in this
                canonical region (e.g. ``"London"``).

        Returns:
            List of ``(council_id, heuristic_score)`` tuples sorted
            descending by score.  Scores are in ``[0, 1]``.
        """
        if not council_stats:
            return []

        # ── region filter ─────────────────────────────────────────────
        if region:
            total = len(council_stats)
            filtered = {
                cid: stats
                for cid, stats in council_stats.items()
                if getattr(stats, "region", None) == region
            }
            if filtered:
                logger.info(
                    "Region filter '%s': %d councils (of %d total)",
                    region, len(filtered), total,
                )
                council_stats = filtered
            else:
                logger.warning(
                    "Region '%s' matched 0 councils — falling back to all %d",
                    region, total,
                )

        project_key = _PROJECT_TYPE_MAP.get(intent.project_type, "residential")

        # ── collect raw values for normalisation ─────────────────────
        raw_scores: list[tuple[int, float, float, float, float]] = []
        all_speeds: list[float] = []
        all_volumes: list[float] = []

        for cid, stats in council_stats.items():
            approval = stats.approval_rate or 0.0

            speed = 0.0
            if stats.average_decision_time:
                speed = stats.average_decision_time.get(project_key, 0.0)
            all_speeds.append(speed)

            activity_str = (
                stats.council_development_activity_level or ""
            ).lower()
            activity = _ACTIVITY_SCORES.get(activity_str, 0.0)

            volume = float(stats.number_of_new_homes_approved or 0)
            all_volumes.append(volume)

            raw_scores.append((cid, approval, speed, activity, volume))

        # ── normalise speed and volume ───────────────────────────────
        max_speed = max(all_speeds) if all_speeds else 1.0
        max_volume = max(all_volumes) if all_volumes else 1.0

        scored: list[tuple[int, float]] = []
        for cid, approval, speed, activity, volume in raw_scores:
            norm_speed = 1.0 - (speed / max_speed) if max_speed > 0 else 0.5
            norm_volume = volume / max_volume if max_volume > 0 else 0.0

            score = (
                self._w_approval * approval
                + self._w_speed * norm_speed
                + self._w_activity * activity
                + self._w_volume * norm_volume
            )

            scored.append((cid, round(score, 6)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        logger.info(
            "Heuristic-ranked %d councils for '%s' (project_key=%s), "
            "top score=%.4f",
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
