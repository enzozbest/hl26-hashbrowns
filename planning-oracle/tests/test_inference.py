"""Tests for the inference layer."""

from __future__ import annotations

import pytest

from inference.parser import ParsedProposal, ProposalParser
from inference.pipeline import InferencePipeline, PredictionResult


class TestProposalParser:
    """Test suite for the NLU proposal parser."""

    def test_extract_postcode(self) -> None:
        """Parser should extract a valid UK postcode from text."""
        # TODO: Verify postcode extraction with known inputs
        ...

    def test_extract_units(self) -> None:
        """Parser should extract unit count from text."""
        # TODO: Verify unit extraction from "20 residential units"
        ...

    def test_extract_floor_area(self) -> None:
        """Parser should extract floor area in sqm."""
        # TODO: Verify extraction from "500 sqm gross floor area"
        ...

    def test_parse_returns_parsed_proposal(self) -> None:
        """parse should return a ParsedProposal model."""
        # TODO: Verify return type and populated fields
        ...

    def test_parse_handles_minimal_input(self) -> None:
        """parse should handle text with no extractable fields gracefully."""
        # TODO: Verify no crash on "build something"
        ...


class TestInferencePipeline:
    """Test suite for the end-to-end inference pipeline."""

    def test_predict_returns_prediction_result(self) -> None:
        """predict should return a PredictionResult model."""
        # TODO: Mock all components and verify return type
        ...

    def test_predict_probability_bounded(self) -> None:
        """Approval probability should be in [0, 1]."""
        # TODO: Mock pipeline and verify bounds
        ...

    def test_predict_includes_attributions(self) -> None:
        """Result should include feature attributions."""
        # TODO: Mock SHAP explainer and verify attributions list
        ...


class TestPredictionAPI:
    """Test suite for the FastAPI endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self) -> None:
        """GET /health should return 200 with status ok."""
        # TODO: Use httpx.AsyncClient with app
        ...

    @pytest.mark.asyncio
    async def test_predict_endpoint_validates_input(self) -> None:
        """POST /predict should reject too-short proposal text."""
        # TODO: Verify 422 on short input
        ...

    @pytest.mark.asyncio
    async def test_predict_endpoint_returns_result(self) -> None:
        """POST /predict should return a PredictionResponse."""
        # TODO: Mock pipeline and verify response shape
        ...
