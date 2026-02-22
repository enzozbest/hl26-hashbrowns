"""Human-readable display names for model feature columns.

Maps internal feature names (matching ``_APP_FEATURE_NAMES`` and
``_COUNCIL_FEATURE_NAMES`` in :mod:`inference.pipeline`) to user-facing
labels suitable for API responses and UI display.
"""

from __future__ import annotations

FEATURE_DISPLAY_NAMES: dict[str, str] = {
    # ── Application features ─────────────────────────────────────────
    "num_new_houses": "Number of new houses proposed",
    "gross_internal_area": "Gross internal area",
    "floor_area_gained": "Floor area to be gained",
    "proposed_gross_floor_area": "Proposed gross floor area",
    "num_comments_received": "Number of public comments received",
    "ratio_one_bed": "Proportion of 1-bed units",
    "ratio_two_bed": "Proportion of 2-bed units",
    "ratio_three_bed": "Proportion of 3-bed units",
    "ratio_four_plus_bed": "Proportion of 4+ bed units",
    "affordable_housing_ratio": "Affordable housing ratio",
    "application_month_sin": "Application month (seasonal)",
    "application_month_cos": "Application month (seasonal cosine)",
    "application_year": "Application year",
    # Missingness indicators
    "missing_num_new_houses": "Data not provided: number of new houses",
    "missing_gross_internal_area": "Data not provided: gross internal area",
    "missing_floor_area_gained": "Data not provided: floor area gained",
    "missing_proposed_gross_floor_area": "Data not provided: proposed gross floor area",
    "missing_num_comments_received": "Data not provided: number of comments",
    "missing_unit_mix": "Data not provided: unit mix",
    "missing_affordable_housing_ratio": "Data not provided: affordable housing ratio",
    # ── Council features ─────────────────────────────────────────────
    "overall_approval_rate": "Overall approval rate",
    "activity_level": "Development activity level",
    "log_total_applications": "Total planning applications (log)",
    "residential_proportion": "Proportion of residential applications",
    "log_new_homes_approved": "New homes approved (log)",
    "approval_rate_by_project_type": "Approval rate for this project type",
    "avg_decision_time_by_project_type": "Average decision time for this project type",
    "log_sample_count_by_project_type": "Sample count for this project type (log)",
    "hdt_measurement": "Housing Delivery Test score",
    "has_green_belt": "Green belt constraint",
}


def get_display_name(feature_name: str) -> str:
    """Return the human-readable display name for *feature_name*.

    Falls back to a title-cased version of the feature name if no
    explicit mapping exists.  The ``missing_*`` pattern is handled
    generically when not in the lookup table.
    """
    if feature_name in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature_name]

    # Generic fallback for unmapped missing_* indicators.
    if feature_name.startswith("missing_"):
        field = feature_name[len("missing_"):].replace("_", " ")
        return f"Data not provided: {field}"

    return feature_name.replace("_", " ").title()
