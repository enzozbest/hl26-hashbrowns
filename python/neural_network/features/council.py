"""Council-level feature extraction.

Transforms council statistics (from the API stats endpoint) into numeric
features that can be joined onto individual planning applications by
``council_id``.

When a council has stats for multiple time windows the temporal join
(``merge_to_applications``) selects the window whose ``period_end`` is
closest to — but not after — each application's ``date_received``.

**Empirical Bayes shrinkage** is applied to per-project-type rate features
(approval rate, average decision time) before they enter the model.  A
council with few observations for a given project type is pulled toward the
global mean for that feature, preventing small-sample noise from dominating
the ranking::

    shrunk_rate = (n * raw_rate + k * global_rate) / (n + k)

where *n* is the council's application count for the matching project type,
*k* is a tuneable shrinkage strength (default 20), and *global_rate* is the
cross-council weighted mean learned during ``fit_transform``.

Exposes a scikit-learn-style ``fit_transform`` / ``transform`` interface.
Fitted state (known project types, activity-level mapping, global rates) is
persisted with pickle.
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

_DEFAULT_SHRINKAGE_K: float = 20.0


class CouncilFeatureExtractor:
    """Derive per-council numeric features from ``CouncilStats`` data.

    Features produced (all ``Float64``):

    * ``overall_approval_rate``  — normalised to 0-1
    * ``activity_level_encoded``  — ordinal (low=0, medium=1, high=2)
    * ``total_applications_per_year``
    * ``log_total_applications``
    * ``residential_proportion``
    * ``new_homes_approved_per_year``
    * ``log_new_homes_approved``

    Per-project-type columns (one per known project type, prefixed
    ``avg_dt_`` and ``n_apps_``) are created from the
    ``average_decision_time`` and ``number_of_applications`` dicts so
    that :meth:`merge_to_applications` can pick the correct value for
    each application.

    Parameters:
        shrinkage_k: Strength of Empirical Bayes shrinkage applied to
            per-project-type rate features.  Higher values pull
            small-sample councils more aggressively toward the global
            mean.  Default 20.
    """

    def __init__(self, *, shrinkage_k: float = _DEFAULT_SHRINKAGE_K) -> None:
        self._fitted: bool = False
        self._activity_map: dict[str, int] = dict(_ACTIVITY_MAP)
        self._known_project_types: list[str] = []
        self._shrinkage_k: float = shrinkage_k
        # Learned during fit — used for Empirical Bayes shrinkage.
        self._global_approval_rate: float = 0.0
        self._global_decision_times: dict[str, float] = {}

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted state to *path*."""
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "fitted": self._fitted,
                    "activity_map": self._activity_map,
                    "known_project_types": self._known_project_types,
                    "shrinkage_k": self._shrinkage_k,
                    "global_approval_rate": self._global_approval_rate,
                    "global_decision_times": self._global_decision_times,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> CouncilFeatureExtractor:
        """Restore a previously fitted extractor."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls(shrinkage_k=state.get("shrinkage_k", _DEFAULT_SHRINKAGE_K))
        inst._fitted = state["fitted"]
        inst._activity_map = state["activity_map"]
        inst._known_project_types = state["known_project_types"]
        inst._global_approval_rate = state.get("global_approval_rate", 0.0)
        inst._global_decision_times = state.get("global_decision_times", {})
        return inst

    # ── public API ────────────────────────────────────────────────────

    def fit_transform(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Learn project-type vocabulary and global rates, then return
        council features.

        Args:
            stats_df: DataFrame with columns matching ``CouncilStats``
                fields (snake_case).

        Returns:
            DataFrame keyed by ``(council_id, period_end)`` with numeric
            feature columns.
        """
        self._learn_project_types(stats_df)
        self._learn_global_rates(stats_df)
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

        Then derives the per-project-type columns with Empirical Bayes
        shrinkage:

        * ``approval_rate_by_matching_project_type`` — shrunk toward
          the global approval rate.
        * ``avg_decision_time_by_matching_project_type`` — shrunk
          toward the global decision time for that project type.
        * ``log_sample_count_by_project_type`` — log1p of the
          application count for the matching project type (signals
          how reliable the rate estimates are).

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

        # ── Empirical Bayes shrinkage for per-project-type features ───

        k = float(self._shrinkage_k)

        # Build per-row sample count (n) for matching project type.
        n_expr: pl.Expr = pl.lit(0.0)
        if self._known_project_types and "project_type" in merged.columns:
            for pt in self._known_project_types:
                col_name = f"n_apps_{self._safe_name(pt)}"
                if col_name in merged.columns:
                    n_expr = (
                        pl.when(pl.col("project_type") == pt)
                        .then(pl.col(col_name).fill_null(0.0))
                        .otherwise(n_expr)
                    )

        merged = merged.with_columns(
            n_expr.cast(pl.Float64).alias("_n_matching"),
        )

        # ── Shrunk approval rate ──────────────────────────────────────
        g_rate = self._global_approval_rate
        merged = merged.with_columns(
            (
                (pl.col("_n_matching") * pl.col("overall_approval_rate").fill_null(0.0) + k * g_rate)
                / (pl.col("_n_matching") + k)
            ).alias("approval_rate_by_matching_project_type"),
        )

        # ── Shrunk decision time ─────────────────────────────────────
        if self._known_project_types and "project_type" in merged.columns:
            # Raw decision time for matching project type.
            dt_raw_expr: pl.Expr = pl.lit(0.0)
            dt_global_expr: pl.Expr = pl.lit(0.0)
            for pt in self._known_project_types:
                col_name = f"avg_dt_{self._safe_name(pt)}"
                if col_name in merged.columns:
                    dt_raw_expr = (
                        pl.when(pl.col("project_type") == pt)
                        .then(pl.col(col_name).fill_null(0.0))
                        .otherwise(dt_raw_expr)
                    )
                g_dt = self._global_decision_times.get(pt, 0.0)
                dt_global_expr = (
                    pl.when(pl.col("project_type") == pt)
                    .then(pl.lit(g_dt))
                    .otherwise(dt_global_expr)
                )
            merged = merged.with_columns(
                (
                    (pl.col("_n_matching") * dt_raw_expr + k * dt_global_expr)
                    / (pl.col("_n_matching") + k)
                ).alias("avg_decision_time_by_matching_project_type"),
            )
        else:
            merged = merged.with_columns(
                pl.lit(0.0).alias(
                    "avg_decision_time_by_matching_project_type",
                ),
            )

        # ── Log sample count (reliability signal for the model) ───────
        merged = merged.with_columns(
            pl.col("_n_matching").log1p().alias(
                "log_sample_count_by_project_type",
            ),
        )

        # Clean up temporary column.
        merged = merged.drop("_n_matching")

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

    def _learn_global_rates(self, stats_df: pl.DataFrame) -> None:
        """Compute cross-council global rates for shrinkage targets.

        Learns:

        * ``_global_approval_rate`` — weighted mean of council overall
          approval rates (0–1), weighted by total application count so
          that large councils contribute proportionally more.
        * ``_global_decision_times`` — per-project-type weighted mean
          of average decision times, weighted by each council's
          application count for that project type.
        """
        rows = stats_df.to_dicts()

        # ── Global approval rate (weighted by total applications) ─────
        total_weight = 0.0
        weighted_sum = 0.0
        for row in rows:
            rate = row.get("approval_rate")
            if rate is None:
                continue
            num_apps: dict = row.get("number_of_applications") or {}
            n = sum(num_apps.values()) if num_apps else 0
            weighted_sum += n * float(rate)
            total_weight += n

        self._global_approval_rate = (
            weighted_sum / total_weight if total_weight > 0 else 0.5
        )
        logger.info(
            "Global approval rate: %.4f (from %.0f total applications)",
            self._global_approval_rate, total_weight,
        )

        # ── Global decision times per project type ────────────────────
        self._global_decision_times = {}
        for pt in self._known_project_types:
            wsum = 0.0
            wn = 0.0
            for row in rows:
                avg_dt: dict = row.get("average_decision_time") or {}
                dt_val = avg_dt.get(pt)
                if dt_val is None:
                    continue
                num_apps = row.get("number_of_applications") or {}
                total_n = sum(num_apps.values()) if num_apps else 0
                n = float(self._count_for_project_type(num_apps, pt, total_n))
                wsum += n * dt_val
                wn += n
            self._global_decision_times[pt] = wsum / wn if wn > 0 else 0.0

        logger.info(
            "Global decision times: %s",
            {k: round(v, 1) for k, v in self._global_decision_times.items()},
        )

    def _transform_impl(self, stats_df: pl.DataFrame) -> pl.DataFrame:
        """Core feature computation (row-wise via ``to_dicts``)."""
        rows = stats_df.to_dicts()
        processed: list[dict] = []

        for row in rows:
            out: dict = {
                "council_id": str(row["council_id"]) if row.get("council_id") is not None else "",
                "period_end": row.get("period_end"),
            }

            # Overall approval rate — API returns 0-1 scale.
            raw_rate = row.get("approval_rate")
            if raw_rate is not None:
                out["overall_approval_rate"] = float(raw_rate)
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
            new_homes = float(row.get("number_of_new_homes_approved") or 0)
            out["new_homes_approved_per_year"] = new_homes
            out["log_new_homes_approved"] = float(np.log1p(new_homes))

            # Per-project-type decision times → one column each.
            avg_dt: dict = row.get("average_decision_time") or {}
            for pt in self._known_project_types:
                col_name = f"avg_dt_{self._safe_name(pt)}"
                out[col_name] = avg_dt.get(pt)

            # Per-project-type application counts → one column each.
            # Used by merge_to_applications for shrinkage sample sizes.
            # The num_apps keys are normalised application types (e.g.
            # "full planning application") while project types are e.g.
            # "residential".  We use keyword matching to approximate
            # the per-project-type count.
            for pt in self._known_project_types:
                col_name = f"n_apps_{self._safe_name(pt)}"
                out[col_name] = float(
                    self._count_for_project_type(num_apps, pt, total),
                )

            # Housing Delivery Test — already on ~0-3 scale (0.94 = 94%).
            hdt_raw = row.get("hdt_measurement")
            out["hdt_measurement"] = float(hdt_raw) if hdt_raw is not None else 1.0

            # Green belt constraint.
            out["has_green_belt"] = 1.0 if row.get("has_green_belt") else 0.0

            processed.append(out)

        if not processed:
            return pl.DataFrame()

        result = pl.DataFrame(processed)

        # Ensure period_end is Date type.
        if "period_end" in result.columns and result["period_end"].dtype == pl.Utf8:
            result = result.with_columns(pl.col("period_end").str.to_date())

        return result

    @staticmethod
    def _count_for_project_type(
        num_apps: dict[str, int],
        project_type: str,
        total: int,
    ) -> int:
        """Approximate the application count for a project type.

        ``number_of_applications`` is keyed by *normalised application type*
        (e.g. "full planning application") while ``average_decision_time``
        is keyed by *project type* (e.g. "residential").  These are
        different key spaces, so a direct lookup would always miss.

        Strategy:
        1. Direct key match (unlikely but cheap).
        2. Substring match — check if the project-type name appears in any
           application-type key or vice-versa.
        3. For residential project types use ``_RESIDENTIAL_KEYWORDS``.
        4. Fall back to total application count (conservative — shrinkage
           uses this as *n*, so a larger *n* means less shrinkage).
        """
        pt_lower = project_type.strip().lower()

        # 1. Direct key match.
        if project_type in num_apps:
            return num_apps[project_type]

        # 2/3. Keyword / substring match.
        matched = 0
        for app_type, count in num_apps.items():
            app_lower = app_type.lower()
            # For residential, use the broader keyword set.
            if pt_lower == "residential" or any(
                kw in pt_lower for kw in _RESIDENTIAL_KEYWORDS
            ):
                if any(kw in app_lower for kw in _RESIDENTIAL_KEYWORDS):
                    matched += count
            elif pt_lower in app_lower or app_lower in pt_lower:
                matched += count

        if matched > 0:
            return matched

        # 4. Fallback — assume total (conservative for shrinkage).
        return total

    @staticmethod
    def _safe_name(s: str) -> str:
        """Sanitise a project-type string for use as a column suffix."""
        return s.strip().lower().replace(" ", "_").replace("-", "_")
