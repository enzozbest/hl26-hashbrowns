"""Feature encoder for planning application response objects.

Extracts numeric features from BaseApplicationsSchema or ApplicationsResponse objects
into numpy arrays suitable for scikit-learn, XGBoost, or any ML pipeline.

The normalised_decision field is separated out as the target variable (label),
not included in the feature matrix.

Encoding strategy:
  - Categorical enums → one-hot binary columns (0.0 / 1.0)
  - Numeric fields → used directly (None → 0.0)
  - Dates → decision_duration_days derived feature
  - Lists (appeals, documents) → count
  - Nested models (proposed_unit_mix, proposed_floor_area) → individual fields

Target encoding (binary):
  - Approved → 1, Refused → 0
  - All other decisions are filtered out by encode_batch
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from hashbrowns.ibex.models import (
    BaseApplicationsSchema,
    NormalisedApplicationType,
    NormalisedDecision,
    ProjectType,
)

TARGET_NAMES: list[str] = ["Refused", "Approved"]

_KEEP_DECISIONS = {NormalisedDecision.Approved, NormalisedDecision.Refused}

_APP_TYPE_MEMBERS = [e.value for e in NormalisedApplicationType]
_PROJECT_TYPE_MEMBERS = [e.value for e in ProjectType]

# Pre-compute one-hot column offsets
_APP_TYPE_LOOKUP = {v: i for i, v in enumerate(_APP_TYPE_MEMBERS)}
_PROJECT_TYPE_LOOKUP = {v: i for i, v in enumerate(_PROJECT_TYPE_MEMBERS)}

# Column layout: council_id | app_type one-hot | project_type one-hot | scalars
_N_APP_TYPE = len(_APP_TYPE_MEMBERS)
_N_PROJECT_TYPE = len(_PROJECT_TYPE_MEMBERS)

_SCALAR_NAMES = [
    "decision_duration_days",
    "num_new_houses",
    "num_comments_received",
    "num_appeals",
    "num_documents",
    "total_existing_residential_units",
    "total_proposed_residential_units",
    "proposed_1_bed_units",
    "proposed_2_bed_units",
    "proposed_3_bed_units",
    "proposed_4_plus_bed_units",
    "affordable_housing_units",
    "gross_internal_area_to_add_sqm",
    "existing_gross_floor_area_sqm",
    "proposed_gross_floor_area_sqm",
    "floor_area_to_be_lost_sqm",
    "floor_area_to_be_gained_sqm",
]

FEATURE_NAMES: list[str] = [
    "council_id",
    *[f"app_type_{m}" for m in _APP_TYPE_MEMBERS],
    *[f"project_type_{m}" for m in _PROJECT_TYPE_MEMBERS],
    *_SCALAR_NAMES,
]

_N_FEATURES = len(FEATURE_NAMES)

# Offsets into the feature vector
_OFF_COUNCIL = 0
_OFF_APP_TYPE = 1
_OFF_PROJECT_TYPE = _OFF_APP_TYPE + _N_APP_TYPE
_OFF_SCALARS = _OFF_PROJECT_TYPE + _N_PROJECT_TYPE


def _extract_scalars(r: BaseApplicationsSchema) -> NDArray[np.float64]:
    """Extract the 17 scalar features from a single response into an array."""
    mix = r.proposed_unit_mix
    area = r.proposed_floor_area

    duration = 0.0
    if r.application_date and r.decided_date:
        duration = float((r.decided_date - r.application_date).days)

    return np.array([
        duration,
        float(r.num_new_houses or 0),
        float(r.num_comments_received or 0),
        float(len(r.appeals)) if r.appeals else 0.0,
        float(len(r.document_metadata)) if r.document_metadata else 0.0,
        float(getattr(mix, "total_existing_residential_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "total_proposed_residential_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "proposed_1_bed_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "proposed_2_bed_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "proposed_3_bed_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "proposed_4_plus_bed_units", 0) or 0) if mix else 0.0,
        float(getattr(mix, "affordable_housing_units", 0) or 0) if mix else 0.0,
        float(getattr(area, "gross_internal_area_to_add_sqm", 0) or 0) if area else 0.0,
        float(getattr(area, "existing_gross_floor_area_sqm", 0) or 0) if area else 0.0,
        float(getattr(area, "proposed_gross_floor_area_sqm", 0) or 0) if area else 0.0,
        float(getattr(area, "floor_area_to_be_lost_sqm", 0) or 0) if area else 0.0,
        float(getattr(area, "floor_area_to_be_gained_sqm", 0) or 0) if area else 0.0,
    ], dtype=np.float64)


def encode(response: BaseApplicationsSchema) -> tuple[NDArray[np.float64], int]:
    """Encode a single BaseApplicationsSchema into a feature vector and target label.

    Returns:
        (X, y) where X is a 1-D float64 array of shape (n_features,) and
        y is the integer class index into TARGET_NAMES.
    """
    row = np.zeros(_N_FEATURES, dtype=np.float64)

    # council_id
    row[_OFF_COUNCIL] = float(response.council_id)

    # One-hot: normalised_application_type
    if response.normalised_application_type is not None:
        idx = _APP_TYPE_LOOKUP.get(response.normalised_application_type.value)
        if idx is not None:
            row[_OFF_APP_TYPE + idx] = 1.0

    # One-hot: project_type
    if response.project_type is not None:
        idx = _PROJECT_TYPE_LOOKUP.get(response.project_type.value)
        if idx is not None:
            row[_OFF_PROJECT_TYPE + idx] = 1.0

    # Scalars
    row[_OFF_SCALARS:] = _extract_scalars(response)

    target = 1 if response.normalised_decision == NormalisedDecision.Approved else 0
    return row, target


def encode_batch(
    responses: Sequence[BaseApplicationsSchema],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Encode a batch of responses into feature matrix and target vector.

    Only Approved and Refused decisions are kept; all other decisions
    (Withdrawn, Other, Unknown, Validated) are filtered out.

    Returns:
        (X, y) where X has shape (n_samples, n_features) and y has shape
        (n_samples,) with binary class labels (1 = Approved, 0 = Refused).
    """
    filtered = [r for r in responses if r.normalised_decision in _KEEP_DECISIONS]

    n = len(filtered)
    X = np.zeros((n, _N_FEATURES), dtype=np.float64)
    y = np.empty(n, dtype=np.intp)

    for i, r in enumerate(filtered):
        X[i], y[i] = encode(r)
    print(f'Num of viable samples: ${len(X)}')
    return X, y
