"""Tests for the SearchResponse feature encoder."""
from datetime import date

import numpy as np
import pytest

from hashbrowns.ibex.models import NormalisedDecision, SearchResponse
from hashbrowns.ml.encoder import (
    FEATURE_NAMES,
    TARGET_NAMES,
    encode,
    encode_batch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_BASE = {
    "council_id": 1,
    "council_name": "Test Council",
    "planning_reference": "2025/001/P",
    "url": "https://example.com",
    "normalised_decision": "Approved",
    "geometry": "POINT(0 0)",
}


def _minimal() -> dict:
    return dict(MINIMAL_BASE)


def _full() -> dict:
    d = dict(MINIMAL_BASE)
    d.update(
        {
            "normalised_application_type": "full planning application",
            "application_date": "2025-04-01",
            "decided_date": "2025-05-08",
            "project_type": "home improvement",
            "num_new_houses": 2,
            "num_comments_received": 3,
            "appeals": [
                {
                    "appeal_ref": "APP/001",
                    "appeal_url": "https://example.com/appeal",
                },
                {
                    "appeal_ref": "APP/002",
                    "appeal_url": "https://example.com/appeal2",
                },
            ],
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
                "total_proposed_residential_units": 4,
                "proposed_1_bed_units": 1,
                "proposed_2_bed_units": 2,
                "proposed_3_bed_units": 1,
                "proposed_4_plus_bed_units": 0,
                "affordable_housing_units": 1,
            },
            "proposed_floor_area": {
                "gross_internal_area_to_add_sqm": 15.5,
                "existing_gross_floor_area_sqm": 85.0,
                "proposed_gross_floor_area_sqm": 100.5,
                "floor_area_to_be_lost_sqm": 0.0,
                "floor_area_to_be_gained_sqm": 15.5,
            },
        }
    )
    return d


def _feat(name: str, row: np.ndarray) -> float:
    """Look up a feature value by name from a 1-D feature vector."""
    return float(row[FEATURE_NAMES.index(name)])


# ---------------------------------------------------------------------------
# Tests — encode returns (ndarray, int)
# ---------------------------------------------------------------------------


class TestEncodeMinimal:
    """Encoding a minimal SearchResponse (only required fields)."""

    def test_returns_tuple(self):
        r = SearchResponse.model_validate(_minimal())
        result = encode(r)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_features_are_ndarray(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float64
        assert X.ndim == 1

    def test_feature_length(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert len(X) == len(FEATURE_NAMES)

    def test_target_is_int(self):
        r = SearchResponse.model_validate(_minimal())
        _, target = encode(r)
        assert isinstance(target, (int, np.integer))

    def test_target_approved(self):
        r = SearchResponse.model_validate(_minimal())
        _, target = encode(r)
        assert target == 1

    def test_decision_onehot_not_in_features(self):
        for member in NormalisedDecision:
            assert f"decision_{member.value}" not in FEATURE_NAMES

    def test_council_id(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert _feat("council_id", X) == 1.0

    def test_optional_enums_all_zero_when_absent(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        for name in FEATURE_NAMES:
            if name.startswith("app_type_") or name.startswith("project_type_"):
                assert _feat(name, X) == 0.0

    def test_numeric_defaults_to_zero(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert _feat("num_new_houses", X) == 0.0
        assert _feat("num_comments_received", X) == 0.0
        assert _feat("num_appeals", X) == 0.0
        assert _feat("num_documents", X) == 0.0
        assert _feat("decision_duration_days", X) == 0.0

    def test_unit_mix_defaults_to_zero(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert _feat("total_proposed_residential_units", X) == 0.0
        assert _feat("proposed_1_bed_units", X) == 0.0
        assert _feat("affordable_housing_units", X) == 0.0

    def test_floor_area_defaults_to_zero(self):
        r = SearchResponse.model_validate(_minimal())
        X, _ = encode(r)
        assert _feat("gross_internal_area_to_add_sqm", X) == 0.0
        assert _feat("existing_gross_floor_area_sqm", X) == 0.0


class TestEncodeFull:
    """Encoding a full SearchResponse with all extension fields populated."""

    def test_application_type_onehot(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("app_type_full planning application", X) == 1.0
        assert _feat("app_type_change of use", X) == 0.0

    def test_project_type_onehot(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("project_type_home improvement", X) == 1.0
        assert _feat("project_type_tree", X) == 0.0

    def test_decision_duration(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        expected = (date(2025, 5, 8) - date(2025, 4, 1)).days
        assert _feat("decision_duration_days", X) == float(expected)

    def test_scalar_numerics(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("num_new_houses", X) == 2.0
        assert _feat("num_comments_received", X) == 3.0

    def test_list_counts(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("num_appeals", X) == 2.0
        assert _feat("num_documents", X) == 1.0

    def test_unit_mix_values(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("total_existing_residential_units", X) == 0.0
        assert _feat("total_proposed_residential_units", X) == 4.0
        assert _feat("proposed_1_bed_units", X) == 1.0
        assert _feat("proposed_2_bed_units", X) == 2.0
        assert _feat("proposed_3_bed_units", X) == 1.0
        assert _feat("proposed_4_plus_bed_units", X) == 0.0
        assert _feat("affordable_housing_units", X) == 1.0

    def test_floor_area_values(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert _feat("gross_internal_area_to_add_sqm", X) == 15.5
        assert _feat("existing_gross_floor_area_sqm", X) == 85.0
        assert _feat("proposed_gross_floor_area_sqm", X) == 100.5
        assert _feat("floor_area_to_be_lost_sqm", X) == 0.0
        assert _feat("floor_area_to_be_gained_sqm", X) == 15.5


class TestTargetEncoding:
    """Target label: Approved → 1, Refused → 0."""

    def test_approved_is_1(self):
        d = _minimal()
        d["normalised_decision"] = "Approved"
        r = SearchResponse.model_validate(d)
        _, target = encode(r)
        assert target == 1

    def test_refused_is_0(self):
        d = _minimal()
        d["normalised_decision"] = "Refused"
        r = SearchResponse.model_validate(d)
        _, target = encode(r)
        assert target == 0

    def test_target_names(self):
        assert TARGET_NAMES == ["Refused", "Approved"]


class TestEncodeBatch:
    """encode_batch filters to Approved/Refused and returns (X, y)."""

    def test_shapes(self):
        responses = [
            SearchResponse.model_validate(_minimal()),  # Approved
            SearchResponse.model_validate(_full()),      # Approved
        ]
        X, y = encode_batch(responses)
        assert X.shape == (2, len(FEATURE_NAMES))
        assert y.shape == (2,)

    def test_dtypes(self):
        responses = [SearchResponse.model_validate(_minimal())]
        X, y = encode_batch(responses)
        assert X.dtype == np.float64
        assert np.issubdtype(y.dtype, np.integer)

    def test_values_match_single_encode(self):
        responses = [
            SearchResponse.model_validate(_minimal()),
            SearchResponse.model_validate(_full()),
        ]
        X, y = encode_batch(responses)
        for i, r in enumerate(responses):
            X_single, y_single = encode(r)
            np.testing.assert_array_equal(X[i], X_single)
            assert y[i] == y_single

    def test_empty_batch(self):
        X, y = encode_batch([])
        assert X.shape == (0, len(FEATURE_NAMES))
        assert y.shape == (0,)

    def test_mixed_decisions(self):
        d1 = _minimal()
        d1["normalised_decision"] = "Approved"
        d2 = _minimal()
        d2["normalised_decision"] = "Refused"
        responses = [
            SearchResponse.model_validate(d1),
            SearchResponse.model_validate(d2),
        ]
        _, y = encode_batch(responses)
        assert y[0] == 1  # Approved
        assert y[1] == 0  # Refused

    def test_filters_out_non_approved_refused(self):
        """Withdrawn, Other, Unknown, Validated are excluded from the batch."""
        decisions = ["Approved", "Refused", "Withdrawn", "Other", "Unknown", "Validated"]
        responses = []
        for dec in decisions:
            d = _minimal()
            d["normalised_decision"] = dec
            responses.append(SearchResponse.model_validate(d))

        X, y = encode_batch(responses)
        assert X.shape[0] == 2  # only Approved + Refused kept
        assert y[0] == 1  # Approved
        assert y[1] == 0  # Refused

    def test_all_filtered_gives_empty(self):
        """If every response is Withdrawn/Other/etc, result is empty."""
        d = _minimal()
        d["normalised_decision"] = "Withdrawn"
        responses = [SearchResponse.model_validate(d)]
        X, y = encode_batch(responses)
        assert X.shape == (0, len(FEATURE_NAMES))
        assert y.shape == (0,)


class TestFeatureNames:
    """FEATURE_NAMES is consistent with encode output."""

    def test_encode_length_matches_feature_names(self):
        r = SearchResponse.model_validate(_full())
        X, _ = encode(r)
        assert len(X) == len(FEATURE_NAMES)

    def test_feature_names_no_duplicates(self):
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_no_decision_onehot_in_feature_names(self):
        for member in NormalisedDecision:
            assert f"decision_{member.value}" not in FEATURE_NAMES
