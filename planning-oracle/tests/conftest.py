"""Shared pytest fixtures for the Planning Oracle test suite."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from config.settings import Settings
from data.schema import (
    CouncilStats,
    DocumentMetadata,
    PlanningApplication,
    ProposedFloorArea,
    ProposedUnitMix,
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
        application_id="APP/2024/001",
        council_id="council-01",
        planning_reference="24/00123/FUL",
        council_name="Test Borough Council",
        description="Erection of 10 residential dwellings with parking.",
        address="1 Test Street",
        postcode="SW1A 1AA",
        ward="Central",
        application_type="Full",
        normalised_application_type="full",
        project_type="residential",
        status="Decided",
        decision="Approved",
        normalised_decision="Approved",
        date_received=date(2024, 3, 15),
        date_validated=date(2024, 3, 20),
        decision_date=date(2024, 6, 10),
        num_new_houses=10,
        proposed_units=ProposedUnitMix(
            one_bed=2, two_bed=4, three_bed=3, four_plus_bed=1, affordable=3,
        ),
        proposed_floor_area=ProposedFloorArea(
            gross_sqm=850.0, net_sqm=720.0, use_class="C3",
        ),
        documents=[
            DocumentMetadata(
                document_id="DOC-001",
                title="Design & Access Statement",
                document_type="DAS",
                url="https://example.com/doc/001",
            ),
        ],
    )


@pytest.fixture
def sample_council_stats() -> CouncilStats:
    """Return sample CouncilStats for testing."""
    return CouncilStats(
        council_id="council-01",
        council_name="Test Borough Council",
        approval_rate=0.72,
        refusal_rate=0.28,
        average_decision_time={"residential": 65.3, "commercial": 45.0},
        number_of_applications={
            "Full": 300,
            "Outline": 120,
            "Reserved Matters": 80,
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
        application_id="APP/2024/001",
        council_id="council-01",
        planning_reference="24/00001/FUL",
        normalised_application_type="full",
        project_type="residential",
        normalised_decision="Approved",
        date_received=date(2024, 3, 15),
        num_new_houses=10,
        proposed_units=ProposedUnitMix(
            one_bed=2, two_bed=4, three_bed=3, four_plus_bed=1, affordable=3,
        ),
        proposed_floor_area=ProposedFloorArea(gross_sqm=850.0, net_sqm=720.0),
    )
    refused = PlanningApplication(
        application_id="APP/2024/002",
        council_id="council-01",
        planning_reference="24/00002/OUT",
        normalised_application_type="outline",
        project_type="commercial",
        normalised_decision="Refused",
        date_received=date(2024, 7, 1),
        num_new_houses=0,
    )
    pending = PlanningApplication(
        application_id="APP/2024/003",
        council_id="council-02",
        planning_reference="24/00003/FUL",
        normalised_application_type="full",
        project_type="residential",
        normalised_decision="Pending",
        date_received=date(2024, 9, 10),
        num_new_houses=5,
    )
    records = (
        [approved.model_dump() for _ in range(10)]
        + [refused.model_dump() for _ in range(8)]
        + [pending.model_dump() for _ in range(2)]
    )
    return pl.DataFrame(records)


@pytest.fixture
def sample_council_stats_df(
    sample_council_stats: CouncilStats,
) -> pl.DataFrame:
    """Return a Polars DataFrame of council stats (possibly multiple windows)."""
    stats_2023 = sample_council_stats.model_dump()
    stats_2022 = dict(
        stats_2023,
        approval_rate=0.68,
        period_start=date(2022, 1, 1),
        period_end=date(2022, 12, 31),
        number_of_new_homes_approved=900,
    )
    return pl.DataFrame([stats_2022, stats_2023])
