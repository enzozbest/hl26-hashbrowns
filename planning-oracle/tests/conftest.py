"""Shared pytest fixtures for the Planning Oracle test suite."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from config.settings import Settings
from data.schema import (
    CouncilStats,
    PlanningApplication,
    ProposedFloorArea,
    ProposedUnitMix,
    SearchDocumentMetadata,
)


@pytest.fixture
def test_settings() -> Settings:
    """Return a Settings instance with test-safe defaults."""
    return Settings(
        planning_api_base_url="https://test.example.com/v1",
        planning_api_auth_token="test-token",
        learning_rate=0.01,
        batch_size=8,
        epochs=2,
        embedding_dim=32,
        text_encoder_model="all-MiniLM-L6-v2",
    )


@pytest.fixture
def sample_application() -> PlanningApplication:
    """Return a minimal PlanningApplication for testing."""
    return PlanningApplication(
        council_id=1,
        planning_reference="24/00123/FUL",
        council_name="Test Borough Council",
        proposal="Erection of 10 residential dwellings with parking.",
        raw_address="1 Test Street, SW1A 1AA",
        raw_application_type="Full",
        normalised_application_type="full planning application",
        project_type="residential",
        raw_decision="Approved",
        normalised_decision="Approved",
        application_date=date(2024, 3, 15),
        decided_date=date(2024, 6, 10),
        num_new_houses=10,
        num_comments_received=5,
        proposed_unit_mix=ProposedUnitMix(
            one_bed=2, two_bed=4, three_bed=3, four_plus_bed=1, affordable=3,
        ),
        proposed_floor_area=ProposedFloorArea(
            gross_internal_area_to_add_sqm=850.0,
            existing_gross_floor_area_sqm=0.0,
            proposed_gross_floor_area_sqm=850.0,
            floor_area_to_be_lost_sqm=0.0,
            floor_area_to_be_gained_sqm=850.0,
        ),
        document_metadata=[
            SearchDocumentMetadata(
                document_type="APPLICATION_FORM",
                description="Application Form",
                document_link="https://example.com/doc/001",
                date_published="2024-03-15",
            ),
        ],
    )


@pytest.fixture
def sample_council_stats() -> CouncilStats:
    """Return sample CouncilStats for testing.

    Note: approval_rate is 0-100 as returned by the API.
    """
    return CouncilStats(
        council_id=1,
        council_name="Test Borough Council",
        approval_rate=72.0,
        refusal_rate=28.0,
        average_decision_time={"residential": 65.3, "commercial": 45.0},
        number_of_applications={
            "full planning application": 300,
            "householder planning application": 120,
            "lawful development": 80,
        },
        number_of_new_homes_approved=1200,
        council_development_activity_level="high",
        period_start=date(2023, 1, 1),
        period_end=date(2023, 12, 31),
    )


@pytest.fixture
def sample_features() -> np.ndarray:
    """Return a small random feature matrix for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((16, 32)).astype(np.float32)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Return a small random binary label vector for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2, size=16).astype(np.float32)


@pytest.fixture
def sample_applications_df() -> pl.DataFrame:
    """Return a Polars DataFrame with a mix of Approved/Refused applications."""
    approved = PlanningApplication(
        council_id=1,
        planning_reference="24/00001/FUL",
        normalised_application_type="full planning application",
        project_type="residential",
        normalised_decision="Approved",
        application_date=date(2024, 3, 15),
        proposal="Erection of 10 dwellings",
        num_new_houses=10,
        proposed_unit_mix=ProposedUnitMix(
            one_bed=2, two_bed=4, three_bed=3, four_plus_bed=1, affordable=3,
        ),
        proposed_floor_area=ProposedFloorArea(
            gross_internal_area_to_add_sqm=850.0,
            proposed_gross_floor_area_sqm=850.0,
            floor_area_to_be_gained_sqm=850.0,
        ),
    )
    refused = PlanningApplication(
        council_id=1,
        planning_reference="24/00002/OUT",
        normalised_application_type="outline",
        project_type="commercial",
        normalised_decision="Refused",
        application_date=date(2024, 7, 1),
        proposal="Change of use to commercial",
        num_new_houses=0,
    )
    pending = PlanningApplication(
        council_id=2,
        planning_reference="24/00003/FUL",
        normalised_application_type="full planning application",
        project_type="residential",
        normalised_decision="Pending",
        application_date=date(2024, 9, 10),
        proposal="Erection of 5 dwellings",
        num_new_houses=5,
    )
    records = (
        [approved.model_dump(mode="json") for _ in range(10)]
        + [refused.model_dump(mode="json") for _ in range(8)]
        + [pending.model_dump(mode="json") for _ in range(2)]
    )
    return pl.DataFrame(records)


@pytest.fixture
def sample_council_stats_df(
    sample_council_stats: CouncilStats,
) -> pl.DataFrame:
    """Return a Polars DataFrame of council stats (possibly multiple windows)."""
    stats_2023 = sample_council_stats.model_dump(mode="json")
    stats_2022 = dict(
        stats_2023,
        approval_rate=68.0,
        period_start="2022-01-01",
        period_end="2022-12-31",
        number_of_new_homes_approved=900,
    )
    return pl.DataFrame([stats_2022, stats_2023])
