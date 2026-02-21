"""Tests for the feature extraction pipelines."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from features.application import ApplicationFeatureExtractor
from features.council import CouncilFeatureExtractor
from features.text import TextEmbedder


class TestApplicationFeatureExtractor:
    """Test suite for application-level feature extraction."""

    def test_fit_transform_drops_non_decided(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """Rows with normalised_decision not in {Approved, Refused} are dropped."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        # Fixture has 10 Approved + 8 Refused + 2 Pending → 18 rows
        assert len(result) == 18

    def test_approved_label_values(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """The 'approved' column should be 1.0 for Approved, 0.0 for Refused."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        labels = result["approved"].to_list()
        assert set(labels) == {0.0, 1.0}
        assert labels.count(1.0) == 10
        assert labels.count(0.0) == 8

    def test_log1p_columns_present(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """Log-transformed numeric columns should appear in the output."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        assert "num_new_houses_log1p" in result.columns

    def test_unit_mix_ratios_sum_to_one(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """Bedroom-type ratio columns should sum to ≈1 when units exist."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        ratio_cols = [c for c in result.columns if c.startswith("ratio_")]
        # First 10 rows (Approved) have units → ratios sum to 1
        sums = result[:10].select(ratio_cols).sum_horizontal()
        for v in sums.to_list():
            assert abs(v - 1.0) < 1e-6

    def test_cyclical_month_bounded(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """Month sin/cos should be in [-1, 1]."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        for col in ("application_month_sin", "application_month_cos"):
            vals = result[col].to_list()
            assert all(-1.0 <= v <= 1.0 for v in vals)

    def test_one_hot_columns_created(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """One-hot columns for categorical features should be present."""
        ext = ApplicationFeatureExtractor()
        result = ext.fit_transform(sample_applications_df)
        app_type_cols = [
            c for c in result.columns if c.startswith("normalised_application_type_")
        ]
        assert len(app_type_cols) >= 2  # at least "full" and "outline"

    def test_transform_uses_fitted_categories(
        self, sample_applications_df: pl.DataFrame,
    ) -> None:
        """transform should use categories learned during fit_transform."""
        ext = ApplicationFeatureExtractor()
        train = ext.fit_transform(sample_applications_df)
        test = ext.transform(sample_applications_df)
        assert train.columns == test.columns

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform without fit_transform should raise."""
        ext = ApplicationFeatureExtractor()
        with pytest.raises(RuntimeError):
            ext.transform(pl.DataFrame({"council_id": ["c1"]}))

    def test_save_load_roundtrip(
        self, sample_applications_df: pl.DataFrame, tmp_path,
    ) -> None:
        """Saving and loading should preserve fitted state."""
        ext = ApplicationFeatureExtractor()
        original = ext.fit_transform(sample_applications_df)
        ext.save(tmp_path / "app_ext.pkl")

        loaded = ApplicationFeatureExtractor.load(tmp_path / "app_ext.pkl")
        restored = loaded.transform(sample_applications_df)
        assert original.columns == restored.columns


class TestCouncilFeatureExtractor:
    """Test suite for council-level feature extraction."""

    def test_fit_transform_produces_features(
        self, sample_council_stats_df: pl.DataFrame,
    ) -> None:
        """fit_transform should return a DataFrame with expected columns."""
        ext = CouncilFeatureExtractor()
        result = ext.fit_transform(sample_council_stats_df)
        for col in (
            "overall_approval_rate",
            "activity_level_encoded",
            "total_applications_per_year",
            "log_total_applications",
            "residential_proportion",
            "new_homes_approved_per_year",
        ):
            assert col in result.columns, f"Missing column: {col}"

    def test_activity_level_ordinal(
        self, sample_council_stats_df: pl.DataFrame,
    ) -> None:
        """Activity level 'high' should encode to 2.0."""
        ext = CouncilFeatureExtractor()
        result = ext.fit_transform(sample_council_stats_df)
        assert result["activity_level_encoded"].to_list() == [2.0, 2.0]

    def test_log_total_positive(
        self, sample_council_stats_df: pl.DataFrame,
    ) -> None:
        """log_total_applications should be positive when applications exist."""
        ext = CouncilFeatureExtractor()
        result = ext.fit_transform(sample_council_stats_df)
        assert all(v > 0 for v in result["log_total_applications"].to_list())

    def test_per_project_type_columns(
        self, sample_council_stats_df: pl.DataFrame,
    ) -> None:
        """Columns for each known project type decision time should exist."""
        ext = CouncilFeatureExtractor()
        result = ext.fit_transform(sample_council_stats_df)
        assert "avg_dt_residential" in result.columns
        assert "avg_dt_commercial" in result.columns

    def test_merge_to_applications(
        self,
        sample_applications_df: pl.DataFrame,
        sample_council_stats_df: pl.DataFrame,
    ) -> None:
        """merge_to_applications should add council features to the app df."""
        ext = CouncilFeatureExtractor()
        features = ext.fit_transform(sample_council_stats_df)
        merged = ext.merge_to_applications(sample_applications_df, features)
        assert "overall_approval_rate" in merged.columns
        assert "avg_decision_time_by_matching_project_type" in merged.columns
        assert "approval_rate_by_matching_project_type" in merged.columns

    def test_transform_before_fit_raises(self) -> None:
        """Calling transform without fit should raise."""
        ext = CouncilFeatureExtractor()
        with pytest.raises(RuntimeError):
            ext.transform(pl.DataFrame({"council_id": ["c1"]}))

    def test_save_load_roundtrip(
        self, sample_council_stats_df: pl.DataFrame, tmp_path,
    ) -> None:
        """Saving and loading should preserve fitted state."""
        ext = CouncilFeatureExtractor()
        original = ext.fit_transform(sample_council_stats_df)
        ext.save(tmp_path / "council_ext.pkl")

        loaded = CouncilFeatureExtractor.load(tmp_path / "council_ext.pkl")
        restored = loaded.transform(sample_council_stats_df)
        assert original.columns == restored.columns


class TestTextEmbedder:
    """Test suite for the text embedding pipeline."""

    def test_embed_single_returns_1d(self) -> None:
        """embed_single should return a 1-D array."""
        # TODO: Requires sentence-transformers model; mock or skip in CI
        ...

    def test_embed_batch_returns_correct_shape(self) -> None:
        """embed_batch should return (n, dim) array."""
        # TODO: Requires sentence-transformers model; mock or skip in CI
        ...

    def test_cache_avoids_recomputation(self) -> None:
        """Calling embed_batch twice on same texts should hit the cache."""
        # TODO: Verify via mock that encode is only called once
        ...
