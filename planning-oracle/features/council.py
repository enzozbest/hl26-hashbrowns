"""Council-level feature extraction.

Transforms council statistics (from the API stats endpoint) into numeric
features that can be joined onto individual planning applications by
``council_id``.

When a council has stats for multiple time windows the temporal join
(``merge_to_applications``) selects the window whose ``period_end`` is
closest to — but not after — each application's ``date_received``.

Exposes a scikit-learn-style ``fit_transform`` / ``transform`` interface.
Fitted state (known project types, activity-level mapping) is persisted
with pickle.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

_ACTIVITY_MAP: dict[str, int] = {"low": 0, "medium": 1, "high": 2}

# Substrings used to identify residential application types when computing
# the residential proportion from the ``number_of_applications`` dict.
_RESIDENTIAL_KEYWORDS: set[str] = {
    "residential",
    "dwelling",
    "housing",
    "houses",
    "householder",
}


class CouncilFeatureExtractor:
    """Derive per-council numeric features from ``CouncilStats`` data.

    Features produced (all ``Float64``):

    * ``overall_approval_rate``  — normalised to 0-1
    * ``activity_level_encoded``  — ordinal (low=0, medium=1, high=2)
    * ``total_applications_per_year``
    * ``log_total_applications``
    * ``residential_proportion``
    * ``new_homes_approved_per_year``

    Per-project-type columns (one per known project type, prefixed
    ``avg_dt_``) are created from the ``average_decision_time`` dict so
    that :meth:`merge_to_applications` can pick the correct value for
    each application.
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._activity_map: dict[str, int] = dict(_ACTIVITY_MAP)
        self._known_project_types: list[str] = []

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted state to *path*."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fitted": self._fitted,
                    "activity_map": self._activity_map,
                    "known_project_types": self._known_project_types,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> CouncilFeatureExtractor:
        """Restore a previously fitted extractor."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls()
        inst._fitted = state["fitted"]
        inst._activity_map = state["activity_map"]
        inst._known_project_types = state["known_project_types"]
        return inst

    # ── public API ────────────────────────────────────────────────────

    def fit_transform(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Learn project-type vocabulary and return council features.

        Args:
            stats_df: DataFrame with columns matching ``CouncilStats``
                fields (snake_case).

        Returns:
            DataFrame keyed by ``(council_id, period_end)`` with numeric
            feature columns.
        """
        self._learn_project_types(stats_df)
        self._fitted = True
        return self._transform_impl(stats_df)

    def transform(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Apply previously-learned encodings to new council stats.

        Raises:
            RuntimeError: If ``fit_transform`` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform")
        return self._transform_impl(stats_df)

    def merge_to_applications(
        self,
        app_df: pl.DataFrame,
        council_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """Temporal-join council features onto an application DataFrame.

        For each application, selects the council-feature row whose
        ``period_end <= date_received`` is maximised (closest preceding
        window).  If ``period_end`` is null on the council features, a
        regular left join on ``council_id`` is used instead.

        Then derives the two per-project-type columns:

        * ``approval_rate_by_matching_project_type`` — falls back to the
          council's overall rate.
        * ``avg_decision_time_by_matching_project_type`` — looked up from
          the ``avg_dt_<project_type>`` columns.

        Args:
            app_df: Application DataFrame (must contain ``council_id``
                and ``date_received``; ``project_type`` is optional).
            council_features: Output of :meth:`fit_transform` /
                :meth:`transform`.

        Returns:
            *app_df* with council-feature columns appended.
        """
        # Ensure date_received exists (may be aliased from application_date).
        if "date_received" not in app_df.columns:
            if "application_date" in app_df.columns:
                app_df = app_df.with_columns(
                    pl.col("application_date").alias("date_received"),
                )
            else:
                raise ValueError(
                    "app_df must contain a 'date_received' or 'application_date' column"
                )

        # Cast dates if stored as strings.
        app = app_df.clone()
        feat = council_features.clone()
        if app["date_received"].dtype == pl.Utf8:
            app = app.with_columns(pl.col("date_received").str.to_date())
        if "period_end" in feat.columns and feat["period_end"].dtype == pl.Utf8:
            feat = feat.with_columns(pl.col("period_end").str.to_date())

        # Ensure council_id types match (both as strings).
        if "council_id" in app.columns:
            app = app.with_columns(pl.col("council_id").cast(pl.Utf8))
        if "council_id" in feat.columns:
            feat = feat.with_columns(pl.col("council_id").cast(pl.Utf8))

        # Decide between temporal join_asof and simple left join.
        has_period_end = (
            "period_end" in feat.columns
            and feat["period_end"].null_count() < len(feat)
        )

        if has_period_end:
            # Sort — required by join_asof.
            app = app.sort("date_received")
            feat = feat.sort("period_end")

            merged = app.join_asof(
                feat,
                left_on="date_received",
                right_on="period_end",
                by="council_id",
                strategy="backward",
            )
        else:
            # No temporal info — plain left join on council_id.
            # Drop period_end from feat if present (all nulls) to avoid
            # conflicts.
            drop_cols = [
                c for c in ["period_end"] if c in feat.columns
            ]
            if drop_cols:
                feat = feat.drop(drop_cols)
            merged = app.join(feat, on="council_id", how="left")

        # ── per-project-type approval rate (fallback to overall) ──────
        merged = merged.with_columns(
            pl.col("overall_approval_rate").alias(
                "approval_rate_by_matching_project_type",
            ),
        )

        # ── per-project-type avg decision time ────────────────────────
        if self._known_project_types and "project_type" in merged.columns:
            expr: pl.Expr = pl.lit(None, dtype=pl.Float64)
            for pt in self._known_project_types:
                col_name = f"avg_dt_{self._safe_name(pt)}"
                if col_name in merged.columns:
                    expr = (
                        pl.when(pl.col("project_type") == pt)
                        .then(pl.col(col_name))
                        .otherwise(expr)
                    )
            merged = merged.with_columns(
                expr.alias("avg_decision_time_by_matching_project_type"),
            )
        else:
            merged = merged.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(
                    "avg_decision_time_by_matching_project_type",
                ),
            )

        return merged

    # ── internals ─────────────────────────────────────────────────────

    def _learn_project_types(self, stats_df: pl.DataFrame) -> None:
        """Collect the union of project types from ``average_decision_time``."""
        seen: set[str] = set()
        if "average_decision_time" not in stats_df.columns:
            self._known_project_types = []
            return
        for val in stats_df["average_decision_time"].to_list():
            if isinstance(val, dict):
                seen.update(val.keys())
        self._known_project_types = sorted(seen)

    def _transform_impl(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Core feature computation (row-wise via ``to_dicts``)."""
        rows = stats_df.to_dicts()
        processed: list[dict] = []

        for row in rows:
            out: dict = {
                "council_id": str(row["council_id"]) if row.get("council_id") is not None else "",
                "period_end": row.get("period_end"),
            }

            # Overall approval rate — API returns 0-100, normalise to 0-1.
            raw_rate = row.get("approval_rate")
            if raw_rate is not None:
                out["overall_approval_rate"] = float(raw_rate) / 100.0
            else:
                out["overall_approval_rate"] = 0.0

            # Activity level → ordinal.
            level = (row.get("council_development_activity_level") or "").lower()
            out["activity_level_encoded"] = float(
                self._activity_map.get(level, -1),
            )

            # Total applications.
            num_apps: dict = row.get("number_of_applications") or {}
            total = sum(num_apps.values()) if num_apps else 0
            out["total_applications_per_year"] = float(total)
            out["log_total_applications"] = float(np.log1p(total))

            # Residential proportion.
            res_count = sum(
                v
                for k, v in num_apps.items()
                if any(kw in k.lower() for kw in _RESIDENTIAL_KEYWORDS)
            )
            out["residential_proportion"] = (
                res_count / total if total > 0 else 0.0
            )

            # New homes.
            out["new_homes_approved_per_year"] = float(
                row.get("number_of_new_homes_approved") or 0,
            )

            # Per-project-type decision times → one column each.
            avg_dt: dict = row.get("average_decision_time") or {}
            for pt in self._known_project_types:
                col_name = f"avg_dt_{self._safe_name(pt)}"
                out[col_name] = avg_dt.get(pt)

            processed.append(out)

        if not processed:
            return pl.DataFrame()

        result = pl.DataFrame(processed)

        # Ensure period_end is Date type.
        if "period_end" in result.columns and result["period_end"].dtype == pl.Utf8:
            result = result.with_columns(pl.col("period_end").str.to_date())

        return result

    @staticmethod
    def _safe_name(s: str) -> str:
        """Sanitise a project-type string for use as a column suffix."""
        return s.strip().lower().replace(" ", "_").replace("-", "_")
