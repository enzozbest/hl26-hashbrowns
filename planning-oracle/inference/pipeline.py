"""End-to-end inference orchestrator.

Coordinates NLU parsing, feature engineering, model prediction, calibration,
council ranking, and SHAP attribution into a single prediction pipeline.

The council ranking step now runs the full :class:`ApprovalModel` in a
single batched forward pass over *all* councils, producing personalised
approval probabilities.  A separate heuristic shortlist is no longer needed.
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Optional

import numpy as np
import torch
from pydantic import BaseModel, Field

from config.feature_display_names import get_display_name
from config.settings import Settings, get_settings
from data.schema import CouncilStats
from features.application import ApplicationFeatureExtractor
from features.council import CouncilFeatureExtractor
from features.text import TextEmbedder
from inference.parser import ProposalIntent, ProposalParser
from model.approval_model import ApprovalModel
from model.calibration import TemperatureScaler
from data.regions import normalise_region_name
from model.council_ranker import CouncilRanker

logger = logging.getLogger(__name__)


# ── Response models ──────────────────────────────────────────────────────────


class FeatureIndicator(BaseModel):
    """A single input feature that influenced a borough's score."""

    name: str = Field(..., description="Feature name (e.g. 'approval_rate')")
    display_name: str = Field("", description="Human-readable feature label")
    value: float = Field(..., description="Raw feature value used by the model")
    contribution: float = Field(
        ...,
        description="Signed magnitude of this feature's contribution to the score",
    )
    direction: str = Field(
        ..., description="'positive' or 'negative' contributor",
    )
    varies_across_councils: Optional[bool] = Field(
        None,
        description="Whether this feature differs across top-ranked councils",
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
        2. Build shared feature tensors (text embedding, application).
        3. Rank *all* councils via the model in a single batched pass.
        4. Compute per-council feature attributions for the top results.
        5. Assemble and return the prediction result.

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

        # ── 2. Build shared feature tensors ──────────────────────────
        app_features = self._build_app_features(intent)
        text_embedding = self._build_text_embedding(intent.raw_text)

        # ── 3. Council ranking (scoped to region when available) ─────
        canonical_region = normalise_region_name(intent.region)
        if canonical_region:
            logger.info("Region scoped: '%s' → '%s'", intent.region, canonical_region)

        # Filter council stats to the target region before ranking.
        scoped_stats = self._council_stats
        if canonical_region:
            filtered = {
                cid: stats
                for cid, stats in self._council_stats.items()
                if getattr(stats, "region", None) == canonical_region
            }
            if filtered:
                logger.info(
                    "Region filter '%s': %d councils (of %d total)",
                    canonical_region, len(filtered), len(self._council_stats),
                )
                scoped_stats = filtered
            else:
                logger.warning(
                    "Region '%s' matched 0 councils — falling back to all %d",
                    canonical_region, len(self._council_stats),
                )

        ranked = self._ranker.rank_councils_with_model(
            intent,
            scoped_stats,
            model=self._model,
            text_embedding=text_embedding,
            app_features=app_features,
            council_feature_builder=self._build_council_feature_vector,
            scaler=self._calibrator,
            top_k=15,
            device=self._device,
        )

        # Warn if any ranked council is outside the expected region.
        if canonical_region:
            for entry in ranked:
                cid = entry["council_id"]
                stats = self._council_stats.get(cid)
                council_region = getattr(stats, "region", None) if stats else None
                if council_region and council_region != canonical_region:
                    logger.warning(
                        "Council %d (%s) region '%s' differs from target '%s'",
                        cid, entry.get("council_name", ""), council_region, canonical_region,
                    )

        # ── 4. Compute attributions for top councils ─────────────────
        top_councils = []
        for entry in ranked:
            cid = entry["council_id"]
            prob = entry["approval_probability"]
            name = entry.get("council_name") or f"Council {cid}"

            council_features = self._build_council_features(cid, intent)
            indicators = self._compute_nn_indicators(
                text_embedding, app_features, council_features,
            )

            top_councils.append(
                CouncilResult(
                    council_id=cid,
                    council_name=name,
                    score=prob,
                    indicators=indicators,
                ),
            )

        # ── 5. Compute varies_across_councils ─────────────────────────
        # Collect each feature's values across all ranked councils.
        if len(top_councils) > 1:
            feature_values: dict[str, set[float]] = {}
            for council in top_councils:
                for ind in council.indicators:
                    feature_values.setdefault(ind.name, set()).add(ind.value)
            for council in top_councils:
                for ind in council.indicators:
                    distinct = feature_values.get(ind.name, set())
                    ind.varies_across_councils = len(distinct) > 1

        # ── 6. Assemble result ───────────────────────────────────────
        # Overall probability comes from the top-ranked council.
        overall_prob = ranked[0]["approval_probability"] if ranked else 0.5
        overall_prob = max(0.0, min(1.0, overall_prob))

        ci = self._confidence_interval(overall_prob, n_approx=100)

        # ── 7. Aggregate feature attributions ranked by impact ────────
        # Use the top council's indicators (matches overall_prob source),
        # sorted by absolute contribution with both positive and negative.
        feature_attributions = self._ranked_feature_attributions(
            top_councils[0].indicators if top_councils else [],
        )

        return PredictionResult(
            parsed_proposal=intent,
            approval_probability=round(overall_prob, 4),
            confidence_interval=(round(ci[0], 4), round(ci[1], 4)),
            top_councils=top_councils,
            feature_attributions=feature_attributions,
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
        # Missingness indicators.
        "missing_num_new_houses",
        "missing_gross_internal_area",
        "missing_floor_area_gained",
        "missing_proposed_gross_floor_area",
        "missing_num_comments_received",
        "missing_unit_mix",
        "missing_affordable_housing_ratio",
    ]

    # Council branch feature names (order must match _COUNCIL_FEATURE_COLS
    # in training/dataset.py).
    _COUNCIL_FEATURE_NAMES: list[str] = [
        "overall_approval_rate",
        "activity_level",
        "log_total_applications",
        "residential_proportion",
        "log_new_homes_approved",
        "approval_rate_by_project_type",
        "avg_decision_time_by_project_type",
        "log_sample_count_by_project_type",
        "hdt_measurement",
        "has_green_belt",
    ]

    def _compute_nn_indicators(
        self,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_features: torch.Tensor,
        *,
        include_zero_contributions: bool = False,
    ) -> list[FeatureIndicator]:
        """Compute gradient×input attributions from the neural network.

        Returns all structured features (application + council branches)
        ranked by absolute contribution.  Every input feature the model
        used is included so consumers can decide how many to show.

        Args:
            text_embedding: ``(1, text_embed_dim)``
            app_features: ``(1, num_app_features)``
            council_features: ``(1, num_council_features)``

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
                display_name=get_display_name(name),
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
                display_name=get_display_name(name),
                value=round(council_vals[i], 6),
                contribution=round(attr_val, 6),
                direction="positive" if attr_val >= 0 else "negative",
            ))

        # Sort by absolute contribution — most influential first.
        indicators.sort(key=lambda x: abs(x.contribution), reverse=True)

        # Filter out zero-contribution features unless explicitly requested.
        if not include_zero_contributions:
            indicators = [ind for ind in indicators if ind.contribution != 0.0]

        return indicators

    @staticmethod
    def _ranked_feature_attributions(
        indicators: list[FeatureIndicator],
    ) -> list[dict[str, Any]]:
        """Build the top-level feature attribution list ranked by impact.

        Returns features sorted by absolute contribution (highest first),
        with both positive and negative contributors clearly labelled.
        Each entry includes a ``rank`` field (1-indexed) for convenience.

        Args:
            indicators: Pre-computed feature indicators (already sorted
                by absolute contribution).

        Returns:
            List of dicts with ``rank``, ``name``, ``display_name``,
            ``value``, ``contribution``, ``direction``, and
            ``varies_across_councils`` keys.
        """
        # Filter out zero-contribution features from the top-level summary.
        nonzero = [ind for ind in indicators if ind.contribution != 0.0]
        return [
            {
                "rank": i + 1,
                "name": ind.name,
                "display_name": ind.display_name,
                "value": ind.value,
                "contribution": ind.contribution,
                "direction": ind.direction,
                "varies_across_councils": ind.varies_across_councils,
            }
            for i, ind in enumerate(nonzero)
        ]

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

        # Year — zeroed out to match the training extractor.  Raw year
        # values cause distribution shift (training saw 2020-2024,
        # inference sees 2026+).  The slot is kept at 0.0 to preserve
        # tensor dimensions.
        features.append(0.0)

        # Missingness indicators (1 = value not provided, 0 = provided).
        # At inference time, only num_houses and unit_mix may be parsed
        # from user input; floor area and comments are never available.
        has_unit_mix = bool(mix and total_beds > 0)
        has_affordable = bool(mix.get("affordable"))
        features.append(0.0 if intent.num_houses is not None else 1.0)  # missing_num_new_houses
        features.append(1.0)                                             # missing_gross_internal_area
        features.append(1.0)                                             # missing_floor_area_gained
        features.append(1.0)                                             # missing_proposed_gross_floor_area
        features.append(1.0)                                             # missing_num_comments_received
        features.append(0.0 if has_unit_mix else 1.0)                    # missing_unit_mix
        features.append(0.0 if has_affordable else 1.0)                  # missing_affordable_housing_ratio

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

    # Map parser project types to council stats keys.
    _PROJECT_TYPE_MAP: dict[str, str] = {
        "small residential": "residential",
        "medium residential": "residential",
        "large residential": "residential",
        "home improvement": "residential",
        "mixed": "commercial",
    }

    def _build_council_feature_vector(
        self,
        stats: CouncilStats,
        intent: ProposalIntent,
    ) -> torch.Tensor:
        """Build a 1-D council feature tensor from stats and intent.

        Satisfies the :class:`CouncilFeatureBuilder` protocol for use
        with :meth:`CouncilRanker.rank_councils_with_model`.

        Returns:
            1-D tensor of shape ``(num_council_features,)``.
        """
        return self._stats_to_council_tensor(stats, intent).squeeze(0)

    def _build_council_features(
        self,
        council_id: Optional[int],
        intent: Optional[ProposalIntent] = None,
    ) -> torch.Tensor:
        """Build council feature tensor for a specific council.

        Used for per-council gradient attribution.
        If no council is available, returns a zero vector.

        Returns:
            Tensor of shape ``(1, num_council_features)``.
        """
        council_dim = self._model.council_branch[0].in_features

        if council_id is None or council_id not in self._council_stats:
            return torch.zeros(1, council_dim)

        return self._stats_to_council_tensor(
            self._council_stats[council_id], intent,
        )

    def _stats_to_council_tensor(
        self,
        stats: CouncilStats,
        intent: Optional[ProposalIntent] = None,
    ) -> torch.Tensor:
        """Convert a :class:`CouncilStats` into a ``(1, D)`` feature tensor.

        Applies Empirical Bayes shrinkage to per-project-type rate
        features using globals learned by the council extractor during
        training.  Uses log1p for count features.

        Feature order matches ``_COUNCIL_FEATURE_COLS`` in
        ``training/dataset.py``.
        """
        council_dim = self._model.council_branch[0].in_features

        # ── Resolve project type ──────────────────────────────────────
        # Use the parser's project type directly — it matches the keys
        # in average_decision_time (e.g. "small residential", "home
        # improvement").  Do NOT map through _PROJECT_TYPE_MAP, which
        # collapses to generic categories ("residential") that don't
        # exist in the council stats dicts.
        pt = "small residential"
        if intent is not None:
            pt = intent.project_type

        # ── Raw stats ─────────────────────────────────────────────────
        approval_rate_norm = stats.approval_rate or 0.0
        num_apps: dict = stats.number_of_applications or {}
        total_apps = sum(num_apps.values()) if num_apps else 0
        # Use the same keyword-matching logic as the training pipeline
        # (CouncilFeatureExtractor._count_for_project_type) to bridge the
        # key-space mismatch: number_of_applications is keyed by
        # application type ("full planning application") while pt is a
        # project type ("small residential").  A naive dict.get() always
        # returns 0.
        n_for_type = float(
            CouncilFeatureExtractor._count_for_project_type(
                num_apps, pt, total_apps,
            ),
        )
        new_homes = float(stats.number_of_new_homes_approved or 0)
        avg_dt_raw = (stats.average_decision_time or {}).get(pt, 0.0)

        # ── Residential proportion ────────────────────────────────────
        res_keywords = {"residential", "dwelling", "housing", "houses", "householder"}
        res_count = sum(
            v for k, v in num_apps.items()
            if any(kw in k.lower() for kw in res_keywords)
        )
        residential_proportion = res_count / total_apps if total_apps > 0 else 0.0

        # ── Empirical Bayes shrinkage ─────────────────────────────────
        k = getattr(self._council_extractor, "_shrinkage_k", 20.0)
        global_rate = getattr(
            self._council_extractor, "_global_approval_rate", approval_rate_norm,
        )
        global_dt = getattr(
            self._council_extractor, "_global_decision_times", {},
        ).get(pt, avg_dt_raw)

        shrunk_rate = (
            (n_for_type * approval_rate_norm + k * global_rate)
            / (n_for_type + k)
        )
        shrunk_dt = (
            (n_for_type * avg_dt_raw + k * global_dt)
            / (n_for_type + k)
        )

        # ── Assemble feature vector (matches _COUNCIL_FEATURE_COLS) ──
        features: list[float] = [
            approval_rate_norm,                              # overall_approval_rate
            {"low": 0.0, "medium": 1.0, "high": 2.0}.get(
                (stats.council_development_activity_level or "").lower(), 0.0,
            ),                                               # activity_level_encoded
            float(np.log1p(total_apps)),                     # log_total_applications
            residential_proportion,                          # residential_proportion
            float(np.log1p(new_homes)),                      # log_new_homes_approved
            shrunk_rate,                                     # approval_rate_by_matching_project_type
            shrunk_dt,                                       # avg_decision_time_by_matching_project_type
            float(np.log1p(n_for_type)),                     # log_sample_count_by_project_type
        ]

        # HDT measurement — already on ~0-3 scale (0.94 = 94%).
        hdt_raw = stats.hdt_measurement
        features.append(float(hdt_raw) if hdt_raw is not None else 1.0)

        # Green belt constraint.
        features.append(1.0 if stats.has_green_belt else 0.0)

        # Pad or truncate to match expected dimension.
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
