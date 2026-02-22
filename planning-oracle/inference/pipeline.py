"""End-to-end inference orchestrator.

Coordinates NLU parsing, feature engineering, model prediction, calibration,
council ranking, and SHAP attribution into a single prediction pipeline.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

from config.settings import Settings, get_settings
from data.schema import CouncilStats
from features.application import ApplicationFeatureExtractor
from features.council import CouncilFeatureExtractor
from features.text import TextEmbedder
from inference.parser import ProposalIntent, ProposalParser
from model.approval_model import ApprovalModel
from model.calibration import TemperatureScaler
from model.council_ranker import CouncilRanker

logger = logging.getLogger(__name__)


# ── Response models ──────────────────────────────────────────────────────────


class FeatureIndicator(BaseModel):
    """A single input feature that influenced a borough's score."""

    name: str = Field(..., description="Feature name (e.g. 'approval_rate')")
    value: float = Field(..., description="Raw feature value used by the model")
    contribution: float = Field(
        ...,
        description="Signed magnitude of this feature's contribution to the score",
    )
    direction: str = Field(
        ..., description="'positive' or 'negative' contributor",
    )


class CouncilResult(BaseModel):
    """A single council in the ranked list."""

    council_id: int
    council_name: Optional[str] = None
    score: float = Field(..., description="Approval affinity score (0-1)")
    indicators: list[FeatureIndicator] = Field(
        default_factory=list,
        description="Ranked list of features that drove this borough's score",
    )


class PredictionResult(BaseModel):
    """Complete prediction response for a user proposal."""

    parsed_proposal: ProposalIntent = Field(
        ..., description="Structured fields extracted from user input",
    )
    approval_probability: float = Field(
        ..., description="Calibrated approval probability (0-1)",
    )
    confidence_interval: tuple[float, float] = Field(
        ..., description="Approximate 95 %% confidence interval",
    )
    top_councils: list[CouncilResult] = Field(
        default_factory=list,
        description="Top councils ranked by approval affinity",
    )
    feature_attributions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top SHAP feature attributions (if explainer available)",
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────


class InferencePipeline:
    """Orchestrate the full prediction pipeline from raw text to result.

    Parameters:
        parser: NLU proposal parser.
        text_embedder: Sentence-transformer text encoder.
        app_extractor: Fitted application feature extractor.
        council_extractor: Fitted council feature extractor.
        council_ranker: Council ranking model.
        model: Trained approval model (eval mode).
        calibrator: Fitted temperature scaler.
        council_stats: Mapping of council_id → CouncilStats.
        settings: Application settings.
    """

    def __init__(
        self,
        parser: ProposalParser,
        text_embedder: TextEmbedder,
        app_extractor: ApplicationFeatureExtractor,
        council_extractor: CouncilFeatureExtractor,
        council_ranker: CouncilRanker,
        model: ApprovalModel,
        calibrator: TemperatureScaler,
        council_stats: dict[int, CouncilStats],
        settings: Optional[Settings] = None,
    ) -> None:
        self._parser = parser
        self._text_embedder = text_embedder
        self._app_extractor = app_extractor
        self._council_extractor = council_extractor
        self._ranker = council_ranker
        self._model = model
        self._calibrator = calibrator
        self._council_stats = council_stats
        self._settings = settings or get_settings()
        self._device = next(model.parameters()).device

    def predict(self, proposal_text: str) -> PredictionResult:
        """Run the full inference pipeline on a free-text proposal.

        Steps:

        1. Parse the proposal into structured fields.
        2. Rank councils by approval affinity.
        3. Build feature tensors (application, council, text).
        4. Predict approval probability via the neural network.
        5. Calibrate the probability.
        6. Assemble and return the prediction result.

        Args:
            proposal_text: Raw user input describing their proposal.

        Returns:
            A complete :class:`PredictionResult`.
        """
        # ── 1. Parse ─────────────────────────────────────────────────
        intent = self._parser.parse(proposal_text)
        logger.info(
            "Parsed: num_houses=%s  project_type=%s  region=%s",
            intent.num_houses, intent.project_type, intent.region,
        )

        # ── 2. Council ranking ───────────────────────────────────────
        ranked = self._ranker.rank_councils(
            intent, self._council_stats, top_k=15,
        )

        # ── 3. Build feature tensors and attributions per council ────
        app_features = self._build_app_features(intent)
        text_embedding = self._build_text_embedding(intent.raw_text)

        top_councils = []
        first_logit = None

        for cid, score, ranker_indicators in ranked:
            stats = self._council_stats.get(cid)
            name = stats.council_name if stats else None

            # Build council-specific features for the neural network.
            council_features = self._build_council_features(cid)

            # Compute gradient-based attributions from the neural net.
            nn_indicators = self._compute_nn_indicators(
                text_embedding, app_features, council_features,
            )

            # Merge ranker indicators (stage-1 scoring components) with
            # neural-net indicators (stage-2 learned feature importance).
            # Ranker indicators come first as they drive the ranking;
            # NN indicators follow as they reflect approval prediction.
            all_indicators = [
                FeatureIndicator(**ind) for ind in ranker_indicators
            ] + nn_indicators

            # Capture the logit from the top-ranked council for the
            # overall approval probability.
            if first_logit is None:
                with torch.no_grad():
                    self._model.eval()
                    first_logit = self._model(
                        text_embedding.to(self._device),
                        app_features.to(self._device),
                        council_features.to(self._device),
                    )

            top_councils.append(
                CouncilResult(
                    council_id=cid,
                    council_name=name,
                    score=score,
                    indicators=all_indicators,
                ),
            )

        # ── 4. Calibrate (using top council's prediction) ────────────
        if first_logit is not None:
            prob = self._calibrator.calibrate(first_logit.cpu()).squeeze().item()
        else:
            prob = 0.5
        prob = max(0.0, min(1.0, prob))

        # Approximate 95% CI using the logistic normal approximation.
        ci = self._confidence_interval(prob, n_approx=100)

        # ── 5. Assemble result ───────────────────────────────────────
        return PredictionResult(
            parsed_proposal=intent,
            approval_probability=round(prob, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            top_councils=top_councils,
            feature_attributions=[],
        )

    # ── Feature name mappings (match order in _build_*_features) ────

    # Application branch feature names (order must match _build_app_features).
    _APP_FEATURE_NAMES: list[str] = [
        "num_new_houses",
        "gross_internal_area",
        "floor_area_gained",
        "proposed_gross_floor_area",
        "num_comments_received",
        "ratio_one_bed",
        "ratio_two_bed",
        "ratio_three_bed",
        "ratio_four_plus_bed",
        "affordable_housing_ratio",
        "application_month_sin",
        "application_month_cos",
        "application_year",
    ]

    # Council branch feature names (order must match _build_council_features).
    _COUNCIL_FEATURE_NAMES: list[str] = [
        "overall_approval_rate",
        "activity_level",
        "total_applications",
        "log_total_applications",
        "residential_proportion",
        "new_homes_approved",
        "approval_rate_by_project_type",
        "avg_decision_time_by_project_type",
    ]

    def _compute_nn_indicators(
        self,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_features: torch.Tensor,
        top_k: int = 6,
    ) -> list[FeatureIndicator]:
        """Compute gradient×input attributions from the neural network.

        Returns the *top_k* most influential structured features (from
        the application and council branches), ranked by absolute
        contribution.

        Args:
            text_embedding: ``(1, text_embed_dim)``
            app_features: ``(1, num_app_features)``
            council_features: ``(1, num_council_features)``
            top_k: Number of top indicators to return.

        Returns:
            List of :class:`FeatureIndicator` sorted by
            ``|contribution|`` descending.
        """
        attrs = self._model.compute_input_attributions(
            text_embedding.to(self._device),
            app_features.to(self._device),
            council_features.to(self._device),
        )

        app_attr = attrs["app"].cpu().tolist()
        council_attr = attrs["council"].cpu().tolist()
        app_vals = app_features.squeeze(0).tolist()
        council_vals = council_features.squeeze(0).tolist()

        # Build named feature list from application branch.
        indicators: list[FeatureIndicator] = []
        for i, attr_val in enumerate(app_attr):
            name = (
                self._APP_FEATURE_NAMES[i]
                if i < len(self._APP_FEATURE_NAMES)
                else f"app_feature_{i}"
            )
            indicators.append(FeatureIndicator(
                name=name,
                value=round(app_vals[i], 6),
                contribution=round(attr_val, 6),
                direction="positive" if attr_val >= 0 else "negative",
            ))

        # Build named feature list from council branch.
        for i, attr_val in enumerate(council_attr):
            name = (
                self._COUNCIL_FEATURE_NAMES[i]
                if i < len(self._COUNCIL_FEATURE_NAMES)
                else f"council_feature_{i}"
            )
            indicators.append(FeatureIndicator(
                name=name,
                value=round(council_vals[i], 6),
                contribution=round(attr_val, 6),
                direction="positive" if attr_val >= 0 else "negative",
            ))

        # Sort by absolute contribution and keep top_k.
        indicators.sort(key=lambda x: abs(x.contribution), reverse=True)
        return indicators[:top_k]

    # ── feature building helpers ─────────────────────────────────────

    def _build_app_features(self, intent: ProposalIntent) -> torch.Tensor:
        """Convert parsed intent into the application feature tensor.

        Mirrors the feature-engineering logic in
        :class:`ApplicationFeatureExtractor` but operates on a single
        synthetic row constructed from the parsed intent.
        """
        num_houses = intent.num_houses or 0

        # Derive unit mix from intent
        mix = intent.unit_mix_preference or {}
        one_bed = mix.get("one_bed", 0)
        two_bed = mix.get("two_bed", 0)
        three_bed = mix.get("three_bed", 0)
        four_plus_bed = mix.get("four_plus_bed", 0)
        total_beds = one_bed + two_bed + three_bed + four_plus_bed

        # If no unit mix was parsed, distribute evenly
        if total_beds == 0 and num_houses > 0:
            two_bed = num_houses // 2
            three_bed = num_houses - two_bed
            total_beds = num_houses

        safe_total = max(total_beds, 1)

        features: list[float] = []

        # Log1p numeric columns (match order in ApplicationFeatureExtractor)
        features.append(float(np.log1p(num_houses)))          # num_new_houses
        features.append(0.0)                                    # gross_internal_area
        features.append(0.0)                                    # floor_area_gained
        features.append(0.0)                                    # proposed_gross_floor
        features.append(0.0)                                    # num_comments

        # Unit-mix ratios
        features.append(one_bed / safe_total)
        features.append(two_bed / safe_total)
        features.append(three_bed / safe_total)
        features.append(four_plus_bed / safe_total)

        # Affordable housing ratio (assume 0 unless specified)
        affordable = mix.get("affordable", 0)
        features.append(affordable / safe_total)

        # Cyclical month (use current month)
        today = date.today()
        month = today.month
        two_pi_over_12 = 2.0 * math.pi / 12.0
        features.append(math.sin(month * two_pi_over_12))
        features.append(math.cos(month * two_pi_over_12))

        # Year
        features.append(float(today.year))

        # One-hot categoricals — pad to match fitted dimensions.
        # We need the fitted category lists from the extractor.
        for col in ("normalised_application_type", "project_type"):
            cats = self._app_extractor._categories.get(col, [])
            # Determine the active category for this input
            if col == "normalised_application_type":
                active = "full"
            else:
                # Map our project_type to the training vocabulary
                pt_map = {
                    "small residential": "residential",
                    "medium residential": "residential",
                    "large residential": "residential",
                    "home improvement": "residential",
                    "mixed": "commercial",
                }
                active = pt_map.get(intent.project_type, "residential")

            for cat in cats:
                features.append(1.0 if cat == active else 0.0)

        # Pad or truncate to match the model's expected input dimension.
        expected_dim = self._model.app_branch[0].in_features
        if len(features) < expected_dim:
            features.extend([0.0] * (expected_dim - len(features)))
        features = features[:expected_dim]

        arr = np.array([features], dtype=np.float32)
        return torch.as_tensor(arr)

    def _build_council_features(
        self, council_id: Optional[int],
    ) -> torch.Tensor:
        """Build council feature tensor for the top-ranked council.

        If no council is available, returns a zero vector.
        """
        # Expected council feature dimension from the model
        council_dim = self._model.council_branch[0].in_features

        if council_id is None or council_id not in self._council_stats:
            return torch.zeros(1, council_dim)

        stats = self._council_stats[council_id]

        # Match the order in _COUNCIL_FEATURE_COLS from dataset.py.
        # API returns approval_rate as 0-100; normalise to 0-1.
        approval_rate_norm = (stats.approval_rate or 0.0) / 100.0
        features: list[float] = [
            approval_rate_norm,
            {"low": 0.0, "medium": 1.0, "high": 2.0}.get(
                (stats.council_development_activity_level or "").lower(), 0.0,
            ),
            float(
                sum((stats.number_of_applications or {}).values()),
            ),
            float(
                np.log1p(
                    sum((stats.number_of_applications or {}).values()),
                ),
            ),
            0.0,  # residential_proportion (would need full computation)
            float(stats.number_of_new_homes_approved or 0),
            approval_rate_norm,  # approval_rate_by_matching_project_type
            0.0,  # avg_decision_time_by_matching_project_type
        ]

        # Pad or truncate to match expected dimension
        if len(features) < council_dim:
            features.extend([0.0] * (council_dim - len(features)))
        features = features[:council_dim]

        arr = np.array([features], dtype=np.float32)
        return torch.as_tensor(arr)

    def _build_text_embedding(self, text: str) -> torch.Tensor:
        """Embed the proposal text using the sentence-transformer."""
        embedding = self._text_embedder.embed_single(text)
        return torch.as_tensor(
            embedding.reshape(1, -1).astype(np.float32),
        )

    @staticmethod
    def _confidence_interval(
        p: float, n_approx: int = 100,
    ) -> tuple[float, float]:
        """Approximate 95 %% Wilson confidence interval.

        Uses the Wilson score interval with *n_approx* as a proxy for
        effective sample size.

        Args:
            p: Point probability estimate.
            n_approx: Effective sample size for interval width.

        Returns:
            ``(lower, upper)``
        """
        z = 1.96
        denom = 1 + z**2 / n_approx
        centre = (p + z**2 / (2 * n_approx)) / denom
        spread = z * math.sqrt(p * (1 - p) / n_approx + z**2 / (4 * n_approx**2)) / denom
        lo = max(0.0, centre - spread)
        hi = min(1.0, centre + spread)
        return lo, hi
