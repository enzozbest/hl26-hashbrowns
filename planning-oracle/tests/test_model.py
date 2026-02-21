"""Tests for the prediction models."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from model.approval_model import ApprovalNet, build_approval_model
from model.calibration import PlattScaler, TemperatureScaler
from model.council_ranker import CouncilRanker


class TestApprovalNet:
    """Test suite for the approval probability network."""

    def test_forward_shape(self, sample_features) -> None:
        """Forward pass should return (batch, 1) tensor."""
        model = ApprovalNet(input_dim=sample_features.shape[1])
        x = torch.tensor(sample_features, dtype=torch.float32)
        out = model(x)
        assert out.shape == (sample_features.shape[0], 1)

    def test_output_bounded(self, sample_features) -> None:
        """Outputs should be in [0, 1] due to sigmoid."""
        model = ApprovalNet(input_dim=sample_features.shape[1])
        x = torch.tensor(sample_features, dtype=torch.float32)
        out = model(x)
        assert (out >= 0).all() and (out <= 1).all()

    def test_build_approval_model_factory(self, test_settings) -> None:
        """build_approval_model should return an ApprovalNet instance."""
        # TODO: Verify factory uses settings correctly
        ...


class TestCalibration:
    """Test suite for probability calibration."""

    def test_temperature_scaler_default(self) -> None:
        """Default temperature should be 1.0 (no scaling)."""
        scaler = TemperatureScaler()
        assert scaler.temperature == 1.0

    def test_platt_scaler_not_fitted(self) -> None:
        """PlattScaler should not be usable before fitting."""
        scaler = PlattScaler()
        assert scaler._scaler is None

    def test_temperature_scaler_fit(
        self, sample_features, sample_labels
    ) -> None:
        """TemperatureScaler.fit should update the temperature parameter."""
        # TODO: Verify temperature changes after fitting
        ...

    def test_platt_scaler_calibrate_shape(
        self, sample_features, sample_labels
    ) -> None:
        """PlattScaler.calibrate should return same-length array."""
        # TODO: Fit and verify output shape
        ...


class TestCouncilRanker:
    """Test suite for the council ranking model."""

    def test_rank_returns_top_k(self) -> None:
        """rank should return exactly top_k results."""
        # TODO: Verify length of rank output
        ...

    def test_rank_sorted_descending(self) -> None:
        """rank results should be sorted by affinity_score descending."""
        # TODO: Verify sort order
        ...
