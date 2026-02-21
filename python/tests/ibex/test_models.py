"""TDD test suite for Ibex Pydantic v2 models.

RED phase: all tests fail because hashbrowns.ibex.models does not exist yet.
GREEN phase: all tests pass after models.py is implemented.
"""
from datetime import date

import pytest
from pydantic import ValidationError

from hashbrowns.ibex.models import (
    AppealsSchema,
    ApplicationsResponse,
    AverageDecisionTime,
    NormalisedDecision,
    ProposedFloorAreaSchema,
    ProposedUnitMixSchema,
    SearchResponse,
    StatsResponse,
)

# ---------------------------------------------------------------------------
# Helpers — minimal dicts for each model
# ---------------------------------------------------------------------------

MINIMAL_BASE = {
    "council_id": 1,
    "council_name": "Test Council",
    "planning_reference": "2025/001/P",
    "url": "https://example.com",
    "normalised_decision": "Approved",
    "geometry": "POINT(0 0)",
}


def _search_minimal() -> dict:
    return dict(MINIMAL_BASE)


def _search_full() -> dict:
    d = dict(MINIMAL_BASE)
    d.update(
        {
            "appeals": [
                {
                    "appeal_ref": "APP/N4720/D/23/3334962",
                    "appeal_url": "https://acp.planninginspectorate.gov.uk/ViewCase.aspx",
                }
            ],
            "project_type": "home improvement",
            "centre_point": "POINT(0 0)",
            "heading": "Double storey extension",
            "num_new_houses": 2,
            "document_metadata": [
                {
                    "date_published": "2025-04-01",
                    "document_type": "APPLICATION_FORM",
                    "description": "Application Form",
                    "document_link": "https://example.com/doc.pdf",
                }
            ],
            "proposed_unit_mix": {
                "total_existing_residential_units": 0,
                "total_proposed_residential_units": 2,
                "proposed_1_bed_units": 1,
                "proposed_2_bed_units": 1,
                "proposed_3_bed_units": 0,
                "proposed_4_plus_bed_units": 0,
                "affordable_housing_units": 0,
            },
            "proposed_floor_area": {
                "gross_internal_area_to_add_sqm": 15.5,
                "existing_gross_floor_area_sqm": 85.0,
                "proposed_gross_floor_area_sqm": 100.5,
                "floor_area_to_be_lost_sqm": 0.0,
                "floor_area_to_be_gained_sqm": 15.5,
            },
            "num_comments_received": 3,
        }
    )
    return d


# ---------------------------------------------------------------------------
# Test 1: minimal SearchResponse — all extension fields default to None
# ---------------------------------------------------------------------------


def test_search_response_minimal():
    """Minimal dict with only required base fields → all 9 extension fields are None."""
    r = SearchResponse.model_validate(_search_minimal())

    assert r.council_id == 1
    assert r.council_name == "Test Council"
    assert r.normalised_decision == NormalisedDecision.Approved

    # All 9 extension fields must default to None when absent
    assert r.appeals is None
    assert r.project_type is None
    assert r.centre_point is None
    assert r.heading is None
    assert r.num_new_houses is None
    assert r.document_metadata is None
    assert r.proposed_unit_mix is None
    assert r.proposed_floor_area is None
    assert r.num_comments_received is None


# ---------------------------------------------------------------------------
# Test 2: full SearchResponse with all extensions populated
# ---------------------------------------------------------------------------


def test_search_response_full_extensions():
    """Dict with all extension fields → each is not None."""
    r = SearchResponse.model_validate(_search_full())

    assert r.appeals is not None
    assert len(r.appeals) == 1
    assert r.appeals[0].appeal_ref == "APP/N4720/D/23/3334962"
    assert r.project_type is not None
    assert r.centre_point is not None
    assert r.heading == "Double storey extension"
    assert r.num_new_houses == 2
    assert r.document_metadata is not None
    assert len(r.document_metadata) == 1
    assert r.proposed_unit_mix is not None
    assert r.proposed_floor_area is not None
    assert r.num_comments_received == 3


# ---------------------------------------------------------------------------
# Test 3: NormalisedDecision enum value
# ---------------------------------------------------------------------------


def test_normalised_decision_approved():
    """NormalisedDecision.Approved string value is 'Approved'."""
    assert NormalisedDecision.Approved == "Approved"


# ---------------------------------------------------------------------------
# Test 4: Invalid normalised_decision raises ValidationError
# ---------------------------------------------------------------------------


def test_normalised_decision_invalid():
    """normalised_decision='INVALID' raises ValidationError."""
    bad = dict(MINIMAL_BASE)
    bad["normalised_decision"] = "INVALID"
    with pytest.raises(ValidationError):
        SearchResponse.model_validate(bad)


# ---------------------------------------------------------------------------
# Test 5: ISO date string parses to date
# ---------------------------------------------------------------------------


def test_date_iso_format():
    """application_date='2025-04-01' parses to date(2025, 4, 1)."""
    d = dict(MINIMAL_BASE)
    d["application_date"] = "2025-04-01"
    r = SearchResponse.model_validate(d)
    assert r.application_date == date(2025, 4, 1)


# ---------------------------------------------------------------------------
# Test 6: Datetime-with-timezone coerces to date
# ---------------------------------------------------------------------------


def test_date_datetime_with_timezone():
    """application_date='2025-04-10T00:00:00.000Z' parses to date(2025, 4, 10)."""
    d = dict(MINIMAL_BASE)
    d["application_date"] = "2025-04-10T00:00:00.000Z"
    r = SearchResponse.model_validate(d)
    assert r.application_date == date(2025, 4, 10)


# ---------------------------------------------------------------------------
# Test 7: Explicit null proposed_unit_mix → None
# ---------------------------------------------------------------------------


def test_extension_explicit_null():
    """proposed_unit_mix=None (explicit null in dict) → None on model."""
    d = dict(MINIMAL_BASE)
    d["proposed_unit_mix"] = None
    r = SearchResponse.model_validate(d)
    assert r.proposed_unit_mix is None


# ---------------------------------------------------------------------------
# Test 8: Absent proposed_unit_mix key → None
# ---------------------------------------------------------------------------


def test_extension_absent():
    """proposed_unit_mix key absent from dict entirely → None on model."""
    r = SearchResponse.model_validate(_search_minimal())
    # Key must not exist in our minimal fixture
    assert "proposed_unit_mix" not in _search_minimal()
    assert r.proposed_unit_mix is None


# ---------------------------------------------------------------------------
# Test 9: ApplicationsResponse has no centre_point field
# ---------------------------------------------------------------------------


def test_applications_response_no_centre_point():
    """ApplicationsResponse does not have a centre_point field."""
    assert not hasattr(ApplicationsResponse.model_fields, "centre_point")
    # Also confirm it doesn't raise even if centre_point is in input (extra=ignore)
    d = dict(MINIMAL_BASE)
    d["centre_point"] = "POINT(1 1)"
    r = ApplicationsResponse.model_validate(d)
    assert not hasattr(r, "centre_point")


# ---------------------------------------------------------------------------
# Test 10: Full StatsResponse validates
# ---------------------------------------------------------------------------


def test_stats_response_valid(stats_response_fixture):
    """Full stats dict validates without error; approval_rate == 85.5."""
    r = StatsResponse.model_validate(stats_response_fixture)
    assert r.approval_rate == 85.5
    assert r.number_of_new_homes_approved == 250


# ---------------------------------------------------------------------------
# Test 11: StatsResponse missing approval_rate raises ValidationError
# ---------------------------------------------------------------------------


def test_stats_response_missing_required(stats_response_fixture):
    """Stats dict missing approval_rate raises ValidationError."""
    bad = dict(stats_response_fixture)
    del bad["approval_rate"]
    with pytest.raises(ValidationError):
        StatsResponse.model_validate(bad)


# ---------------------------------------------------------------------------
# Test 12: AverageDecisionTime handles JSON keys with spaces
# ---------------------------------------------------------------------------


def test_average_decision_time_aliases():
    """AverageDecisionTime parsed from JSON keys with spaces maps to snake_case fields."""
    raw = {
        "small residential": 45.0,
        "tree": 28.0,
        "large residential": 120.0,
        "home improvement": 35.0,
        "mixed": 65.0,
        "medium residential": 75.0,
    }
    adt = AverageDecisionTime.model_validate(raw)
    assert adt.small_residential == 45.0
    assert adt.tree == 28.0
    assert adt.large_residential == 120.0
    assert adt.home_improvement == 35.0
    assert adt.mixed == 65.0
    assert adt.medium_residential == 75.0


# ---------------------------------------------------------------------------
# Test 13: AppealsSchema — required fields only, optionals default to None
# ---------------------------------------------------------------------------


def test_appeals_schema():
    """AppealsSchema with appeal_ref and appeal_url; optional fields all None."""
    a = AppealsSchema.model_validate(
        {
            "appeal_ref": "APP/N4720/D/23/3334962",
            "appeal_url": "https://acp.planninginspectorate.gov.uk/ViewCase.aspx",
        }
    )
    assert a.appeal_ref == "APP/N4720/D/23/3334962"
    assert a.appeal_url == "https://acp.planninginspectorate.gov.uk/ViewCase.aspx"
    assert a.start_date is None
    assert a.decision_date is None
    assert a.decision is None
    assert a.case_type is None


# ---------------------------------------------------------------------------
# Test 14: Extra/unknown fields are ignored (no ValidationError)
# ---------------------------------------------------------------------------


def test_extra_fields_ignored():
    """Passing unknown_field to SearchResponse does not raise ValidationError."""
    d = dict(MINIMAL_BASE)
    d["unknown_field"] = "should be silently ignored"
    r = SearchResponse.model_validate(d)  # must not raise
    assert r.council_id == 1
