"""Application-level feature extraction.

Transforms a Polars DataFrame of planning applications into a numeric
feature matrix suitable for model input.  Exposes a scikit-learn-style
``fit_transform`` / ``transform`` interface so that categorical encodings
learned on the training split are applied consistently to validation and
test splits.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

_TWO_PI_OVER_12 = 2.0 * math.pi / 12.0

# Columns that get a log1p transform.  These match the API response field
# names as they appear on the PlanningApplication and ProposedFloorArea
# models.  If a column is absent from the input DataFrame it is filled
# with 0 before the transform.
_LOG_COLS: list[str] = [
    "num_new_houses",
    "gross_internal_area_to_add_sqm",
    "floor_area_to_be_gained_sqm",
    "proposed_gross_floor_area_sqm",
    "num_comments_received",
]

# Unit-mix bedroom columns (from the ProposedUnitMix model).
_UNIT_COLS: list[str] = ["one_bed", "two_bed", "three_bed", "four_plus_bed"]
_AFFORDABLE_COL = "affordable"

# Categorical columns to one-hot encode.
_CATEGORICAL_COLS: list[str] = ["normalised_application_type", "project_type"]

# Columns carried through as row identifiers (not features).
_INDEX_COLS: list[str] = ["council_id", "planning_reference"]

# Column used to derive the binary label.
_LABEL_COL = "normalised_decision"

# Mapping from ProposedFloorArea struct fields to the expected flat column
# names used in _LOG_COLS.  These now match the actual API field names.
_FLOOR_AREA_FIELDS: list[str] = [
    "gross_internal_area_to_add_sqm",
    "existing_gross_floor_area_sqm",
    "proposed_gross_floor_area_sqm",
    "floor_area_to_be_lost_sqm",
    "floor_area_to_be_gained_sqm",
]


class ApplicationFeatureExtractor:
    """Extract numeric features from a planning-application DataFrame.

    **Fit** learns the unique category sets for one-hot encoding.
    **Transform** applies log1p, cyclical-month, unit-mix ratios,
    one-hot encoding, and the binary approval label.

    The fitted state (category mappings) can be persisted with
    :meth:`save` / :meth:`load` (pickle).
    """

    def __init__(self) -> None:
        self._fitted: bool = False
        self._categories: dict[str, list[str]] = {}

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the fitted state to *path*."""
        with open(path, "wb") as f:
            pickle.dump(
                {"categories": self._categories, "fitted": self._fitted}, f,
            )

    @classmethod
    def load(cls, path: str | Path) -> ApplicationFeatureExtractor:
        """Restore a previously fitted extractor from *path*."""
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls()
        inst._categories = state["categories"]
        inst._fitted = state["fitted"]
        return inst

    # ── public API ────────────────────────────────────────────────────

    def fit_transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Learn encodings from *df* and return the feature matrix.

        Rows whose ``normalised_decision`` is not *Approved* or *Refused*
        are dropped.

        Returns:
            DataFrame with float feature columns, index columns
            (``council_id``, ``planning_reference``), and a binary
            ``approved`` label.
        """
        df = self._prepare(df)
        df = self._filter_decisions(df)
        for col in _CATEGORICAL_COLS:
            if col in df.columns:
                self._categories[col] = (
                    df[col].drop_nulls().unique().sort().to_list()
                )
            else:
                self._categories[col] = []
        self._fitted = True
        return self._build_features(df)

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply previously-learned encodings to *df*.

        Raises:
            RuntimeError: If :meth:`fit_transform` has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform")
        df = self._prepare(df)
        df = self._filter_decisions(df)
        return self._build_features(df)

    # ── internals ─────────────────────────────────────────────────────

    @staticmethod
    def _filter_decisions(df: pl.DataFrame) -> pl.DataFrame:
        """Keep only rows with a definitive Approved / Refused decision."""
        return df.filter(
            pl.col(_LABEL_COL).is_in(["Approved", "Refused"]),
        )

    @staticmethod
    def _prepare(df: pl.DataFrame) -> pl.DataFrame:
        """Flatten struct columns and ensure every expected column exists."""

        # ── unnest proposed_unit_mix struct ─────────────────────────────
        if "proposed_unit_mix" in df.columns:
            col_dtype = df["proposed_unit_mix"].dtype
            is_struct = isinstance(col_dtype, pl.Struct)
            all_null = df["proposed_unit_mix"].null_count() == len(df)
            if is_struct and not all_null:
                for field in _UNIT_COLS + [_AFFORDABLE_COL]:
                    if field not in df.columns:
                        df = df.with_columns(
                            pl.col("proposed_unit_mix")
                            .struct.field(field)
                            .fill_null(0)
                            .alias(field),
                        )
            df = df.drop("proposed_unit_mix")

        # ── unnest proposed_floor_area struct ──────────────────────────
        if "proposed_floor_area" in df.columns:
            col_dtype = df["proposed_floor_area"].dtype
            is_struct = isinstance(col_dtype, pl.Struct)
            all_null = df["proposed_floor_area"].null_count() == len(df)
            if is_struct and not all_null:
                for field in _FLOOR_AREA_FIELDS:
                    if field not in df.columns:
                        df = df.with_columns(
                            pl.col("proposed_floor_area")
                            .struct.field(field)
                            .fill_null(0.0)
                            .alias(field),
                        )
            df = df.drop("proposed_floor_area")

        # ── guarantee numeric columns exist (fill with 0) ────────────
        for col in _LOG_COLS:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

        for col in _UNIT_COLS + [_AFFORDABLE_COL]:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0).alias(col))

        # ── guarantee index columns exist ─────────────────────────────
        if "planning_reference" not in df.columns:
            df = df.with_columns(pl.lit("").alias("planning_reference"))

        # ── ensure council_id is string for index purposes ────────────
        if "council_id" in df.columns:
            df = df.with_columns(pl.col("council_id").cast(pl.Utf8))

        # ── map application_date → date_received if needed ────────────
        if "date_received" not in df.columns and "application_date" in df.columns:
            df = df.with_columns(pl.col("application_date").alias("date_received"))

        # ── map proposal → description if needed ──────────────────────
        if "description" not in df.columns and "proposal" in df.columns:
            df = df.with_columns(pl.col("proposal").alias("description"))

        # ── cast date column if stored as string ──────────────────────
        if "date_received" in df.columns and df["date_received"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("date_received").str.to_date())

        return df

    def _build_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Assemble every feature column in a single ``select``."""
        exprs: list[pl.Expr] = []

        # ── index columns ─────────────────────────────────────────────
        for col in _INDEX_COLS:
            exprs.append(pl.col(col))

        # ── numeric log1p ─────────────────────────────────────────────
        for col in _LOG_COLS:
            exprs.append(
                pl.col(col)
                .cast(pl.Float64)
                .fill_null(0.0)
                .log1p()
                .alias(f"{col}_log1p"),
            )

        # ── unit-mix ratios ───────────────────────────────────────────
        total_units = pl.sum_horizontal(
            [pl.col(c).cast(pl.Float64).fill_null(0.0) for c in _UNIT_COLS],
        )
        safe_total = pl.when(total_units > 0).then(total_units).otherwise(1.0)

        for col in _UNIT_COLS:
            exprs.append(
                (pl.col(col).cast(pl.Float64).fill_null(0.0) / safe_total).alias(
                    f"ratio_{col}",
                ),
            )
        exprs.append(
            (
                pl.col(_AFFORDABLE_COL).cast(pl.Float64).fill_null(0.0) / safe_total
            ).alias("affordable_housing_ratio"),
        )

        # ── temporal (cyclical month + year) ──────────────────────────
        has_date = pl.col("date_received").is_not_null()
        month = pl.col("date_received").dt.month().cast(pl.Float64)
        exprs.extend(
            [
                pl.when(has_date)
                .then((month * _TWO_PI_OVER_12).sin())
                .otherwise(0.0)
                .alias("application_month_sin"),
                pl.when(has_date)
                .then((month * _TWO_PI_OVER_12).cos())
                .otherwise(0.0)
                .alias("application_month_cos"),
                pl.col("date_received")
                .dt.year()
                .cast(pl.Float64)
                .fill_null(0.0)
                .alias("application_year"),
            ],
        )

        # ── categorical one-hot ───────────────────────────────────────
        for col, cats in self._categories.items():
            for cat in cats:
                exprs.append(
                    (pl.col(col).fill_null("") == cat)
                    .cast(pl.Float64)
                    .alias(f"{col}_{cat}"),
                )

        # ── label ─────────────────────────────────────────────────────
        exprs.append(
            (pl.col(_LABEL_COL) == "Approved")
            .cast(pl.Float64)
            .alias("approved"),
        )

        return df.select(exprs)
