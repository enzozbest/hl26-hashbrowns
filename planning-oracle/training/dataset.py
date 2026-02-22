"""PyTorch Dataset and DataLoader with temporal train/val/test splits.

Splits data chronologically to prevent data leakage — the model never sees
future applications during training.  The :class:`PlanningDataset` stores
three feature branches (application, council, text) as separate tensors so
the downstream model can process each branch independently before fusion.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data.api_client import PlanningAPIClient
from data.schema import CouncilStats, PlanningApplication
from features.application import (
    ApplicationFeatureExtractor,
    _CATEGORICAL_COLS,
    _LOG_COLS,
    _UNIT_COLS,
)
from features.council import CouncilFeatureExtractor
from features.text import TextEmbedder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Columns pulled from the council merge output to form the council branch.
_COUNCIL_FEATURE_COLS: list[str] = [
    "overall_approval_rate",
    "activity_level_encoded",
    "total_applications_per_year",
    "log_total_applications",
    "residential_proportion",
    "new_homes_approved_per_year",
    "approval_rate_by_matching_project_type",
    "avg_decision_time_by_matching_project_type",
]

_INDEX_COLS = frozenset({"council_id", "planning_reference"})
_LABEL_COL = "approved"

# Default temporal-split boundaries.
_TRAIN_CUTOFF = date(2024, 1, 1)
_VAL_CUTOFF = date(2025, 1, 1)

# The date column used for temporal splits — matches the API field name.
_DATE_COL = "application_date"


# ── Dataset ──────────────────────────────────────────────────────────────────


class PlanningDataset(Dataset):
    """Multi-branch PyTorch Dataset for the planning approval model.

    Each sample is a dict of four tensors so that the model can route
    each branch through its own sub-network before fusion.

    Parameters:
        app_features:     ``(N, num_app_features)`` structured application
                          features (log-transformed numerics, ratios,
                          cyclical month, one-hot categoricals).
        council_features: ``(N, num_council_features)`` council context
                          features (approval rates, decision speed, etc.).
        text_embeddings:  ``(N, text_embed_dim)`` pre-computed sentence-
                          transformer embeddings of proposal descriptions.
        labels:           ``(N,)`` binary labels (1 = Approved, 0 = Refused).
    """

    def __init__(
        self,
        app_features: np.ndarray,
        council_features: np.ndarray,
        text_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        self.app_features = torch.as_tensor(
            app_features.astype(np.float32), dtype=torch.float32,
        )
        self.council_features = torch.as_tensor(
            council_features.astype(np.float32), dtype=torch.float32,
        )
        self.text_embeddings = torch.as_tensor(
            text_embeddings.astype(np.float32), dtype=torch.float32,
        )
        self.labels = torch.as_tensor(
            labels.astype(np.float32), dtype=torch.float32,
        )

        n = len(self.labels)
        if self.app_features.shape[0] != n:
            raise ValueError(
                f"app_features has {self.app_features.shape[0]} rows, "
                f"expected {n}",
            )
        if self.council_features.shape[0] != n:
            raise ValueError(
                f"council_features has {self.council_features.shape[0]} rows, "
                f"expected {n}",
            )
        if self.text_embeddings.shape[0] != n:
            raise ValueError(
                f"text_embeddings has {self.text_embeddings.shape[0]} rows, "
                f"expected {n}",
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single sample as a dict of tensors.

        Keys: ``app_features``, ``council_features``, ``text_embedding``,
        ``label``.
        """
        return {
            "app_features": self.app_features[idx],
            "council_features": self.council_features[idx],
            "text_embedding": self.text_embeddings[idx],
            "label": self.labels[idx],
        }


# ── Temporal split ───────────────────────────────────────────────────────────


def temporal_split(
    df: pl.DataFrame,
    date_column: str = _DATE_COL,
    *,
    train_cutoff: date = _TRAIN_CUTOFF,
    val_cutoff: date = _VAL_CUTOFF,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split a DataFrame chronologically into train / val / test.

    * **Train**: ``date_column < train_cutoff``
    * **Val**:   ``train_cutoff <= date_column < val_cutoff``
    * **Test**:  ``date_column >= val_cutoff``

    Rows with a null date are dropped.

    Args:
        df: Input DataFrame.
        date_column: Name of the date column to split on.
        train_cutoff: First day of the validation period (exclusive upper
            bound for train).
        val_cutoff: First day of the test period (exclusive upper bound
            for val).

    Returns:
        ``(train_df, val_df, test_df)``
    """
    df = df.filter(pl.col(date_column).is_not_null())

    if df[date_column].dtype == pl.Utf8:
        df = df.with_columns(pl.col(date_column).str.to_date())

    col = pl.col(date_column)
    train = df.filter(col < train_cutoff)
    val = df.filter((col >= train_cutoff) & (col < val_cutoff))
    test = df.filter(col >= val_cutoff)

    logger.info(
        "Temporal split on '%s' (cutoffs %s / %s): "
        "train=%d, val=%d, test=%d",
        date_column,
        train_cutoff,
        val_cutoff,
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


# ── End-to-end dataset construction ──────────────────────────────────────────


async def build_datasets(
    client: PlanningAPIClient,
    text_embedder: TextEmbedder,
    app_extractor: ApplicationFeatureExtractor,
    council_extractor: CouncilFeatureExtractor,
    *,
    council_ids: list[int],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> tuple[PlanningDataset, PlanningDataset, PlanningDataset]:
    """Orchestrate the full feature-engineering pipeline and return datasets.

    Steps:

    1. Fetch applications and council stats from the Planning API.
    2. Filter to decided cases (Approved / Refused only).
    3. Temporal split (train < 2024-01-01, val 2024, test >= 2025-01-01).
    4. Fit ``app_extractor`` and ``council_extractor`` on the **training
       set only**, then transform all three splits.
    5. Compute text embeddings for every proposal description.
    6. Return ``(train_dataset, val_dataset, test_dataset)``.

    Args:
        client: Configured :class:`PlanningAPIClient` for fetching data.
        text_embedder: Initialised :class:`TextEmbedder`.
        app_extractor: Un-fitted :class:`ApplicationFeatureExtractor`
            (will be fitted on the train split).
        council_extractor: Un-fitted :class:`CouncilFeatureExtractor`
            (will be fitted on the council stats).
        council_ids: List of council identifier integers.
        date_from: Optional ISO-8601 start date for the search window.
        date_to: Optional ISO-8601 end date for the search window.
        checkpoint_dir: If provided, save council stats JSON here for
            later use at inference time.

    Returns:
        Three :class:`PlanningDataset` instances (train, val, test).
    """

    # ── 1. Fetch from API ─────────────────────────────────────────────
    logger.info("Fetching applications for %d councils …", len(council_ids))
    all_apps: list[PlanningApplication] = []
    for cid in council_ids:
        apps = await client.search_all_pages(
            str(cid), date_from=date_from, date_to=date_to,
        )
        all_apps.extend(apps)
        logger.info("  %d: %d applications", cid, len(apps))

    logger.info("Fetching council stats …")
    all_stats: list[CouncilStats] = []
    for cid in council_ids:
        stats = await client.get_council_stats(
            cid, date_from=date_from, date_to=date_to,
        )
        all_stats.append(stats)

    # Save council stats as JSON for inference-time use.
    if checkpoint_dir is not None:
        stats_path = Path(checkpoint_dir) / "council_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(
                [s.model_dump(mode="json") for s in all_stats],
                indent=2,
            ),
        )
        logger.info("Council stats saved to %s", stats_path)

    # Convert Pydantic models → Polars DataFrames.
    apps_df = pl.DataFrame([a.model_dump(mode="json") for a in all_apps])
    stats_df = pl.DataFrame([s.model_dump(mode="json") for s in all_stats])

    # ── 2. Pre-filter to decided cases ────────────────────────────────
    apps_df = apps_df.filter(
        pl.col("normalised_decision").is_in(["Approved", "Refused"]),
    )
    logger.info("Decided applications loaded: %d rows", len(apps_df))

    # Add a stable row-index so we can restore the original order after
    # the council merge (which sorts by date for join_asof).
    apps_df = apps_df.with_row_index("_row_idx")

    # ── 3. Temporal split ─────────────────────────────────────────────
    train_raw, val_raw, test_raw = temporal_split(apps_df)

    # ── 4a. Application features (fit on train, transform all) ────────
    train_app_df = app_extractor.fit_transform(train_raw)
    val_app_df = app_extractor.transform(val_raw)
    test_app_df = app_extractor.transform(test_raw)

    # Feature columns = everything output by the extractor minus the
    # index identifiers and the label.
    app_feature_cols = [
        c
        for c in train_app_df.columns
        if c not in _INDEX_COLS and c != _LABEL_COL
    ]

    # ── 4b. Council features (fit on stats, merge to each split) ──────
    council_feat_df = council_extractor.fit_transform(stats_df)

    def _merge_council(raw_split: pl.DataFrame) -> pl.DataFrame:
        merged = council_extractor.merge_to_applications(
            raw_split, council_feat_df,
        )
        # Restore the pre-merge row order so arrays stay aligned with
        # the app features and text embeddings (which preserve order).
        return merged.sort("_row_idx")

    train_council_df = _merge_council(train_raw)
    val_council_df = _merge_council(val_raw)
    test_council_df = _merge_council(test_raw)

    # Only keep council feature columns that actually exist after merge.
    council_cols = [
        c for c in _COUNCIL_FEATURE_COLS if c in train_council_df.columns
    ]

    # ── 5. Text embeddings ────────────────────────────────────────────
    def _embed(raw_split: pl.DataFrame) -> np.ndarray:
        # The API field is "proposal"; _prepare() aliases it to
        # "description", but we read from the raw split here.
        col_name = "proposal" if "proposal" in raw_split.columns else "description"
        texts = raw_split[col_name].fill_null("").to_list()
        return text_embedder.embed_batch(texts)

    logger.info("Computing text embeddings …")
    train_emb = _embed(train_raw)
    val_emb = _embed(val_raw)
    test_emb = _embed(test_raw)

    # ── 6. Assemble PlanningDatasets ──────────────────────────────────
    def _to_dataset(
        app_df: pl.DataFrame,
        council_df: pl.DataFrame,
        emb: np.ndarray,
    ) -> PlanningDataset:
        app_np = (
            app_df.select(app_feature_cols)
            .fill_null(0.0)
            .cast({c: pl.Float64 for c in app_feature_cols})
            .to_numpy()
            .astype(np.float32)
        )
        council_np = (
            council_df.select(council_cols)
            .fill_null(0.0)
            .cast({c: pl.Float64 for c in council_cols})
            .to_numpy()
            .astype(np.float32)
        )
        labels_np = app_df[_LABEL_COL].to_numpy().astype(np.float32)
        return PlanningDataset(app_np, council_np, emb, labels_np)

    train_ds = _to_dataset(train_app_df, train_council_df, train_emb)
    val_ds = _to_dataset(val_app_df, val_council_df, val_emb)
    test_ds = _to_dataset(test_app_df, test_council_df, test_emb)

    logger.info(
        "Datasets ready — train=%d  val=%d  test=%d | "
        "app_dim=%d  council_dim=%d  text_dim=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        train_ds.app_features.shape[1],
        train_ds.council_features.shape[1],
        train_ds.text_embeddings.shape[1],
    )
    return train_ds, val_ds, test_ds


# ── DataLoaders ──────────────────────────────────────────────────────────────


def build_dataloaders(
    train: PlanningDataset,
    val: PlanningDataset,
    test: PlanningDataset,
    batch_size: int = 256,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders with class-balanced sampling for the train split.

    * **Train**: :class:`WeightedRandomSampler` with weights inversely
      proportional to class frequency, ``drop_last=True``.
    * **Val / Test**: sequential iteration, ``shuffle=False``.
    * All loaders use ``num_workers=4`` and ``pin_memory=True`` when
      CUDA is available.

    Args:
        train: Training dataset.
        val:   Validation dataset.
        test:  Test dataset.
        batch_size: Mini-batch size (default 256).

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """
    pin = torch.cuda.is_available()
    workers = 0

    # ── class-balanced sampler for train ──────────────────────────────
    labels_np = train.labels.numpy()
    class_counts = np.bincount(labels_np.astype(int), minlength=2).astype(
        np.float64,
    )
    # Weight each class inversely proportional to its frequency.
    class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    sample_weights = class_weights[labels_np.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train),
        replacement=True,
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        sampler=sampler,        # handles shuffling
        drop_last=True,
        num_workers=workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    logger.info(
        "DataLoaders ready — batch_size=%d  train_batches=%d  "
        "val_batches=%d  test_batches=%d  (pin_memory=%s, workers=%d)",
        batch_size,
        len(train_loader),
        len(val_loader),
        len(test_loader),
        pin,
        workers,
    )
    return train_loader, val_loader, test_loader


# ── Feature-name introspection (for SHAP) ────────────────────────────────────


def get_feature_names(
    app_extractor: ApplicationFeatureExtractor,
    council_extractor: CouncilFeatureExtractor,
    text_embedder: TextEmbedder,
) -> dict[str, list[str]]:
    """Return human-readable feature names grouped by model branch.

    The lists are ordered to match the tensor columns produced by
    :func:`build_datasets`, which is critical for mapping SHAP values
    back to interpretable names.

    All three components must be fitted / initialised before calling.

    Returns:
        ``{"app": [...], "council": [...], "text": [...]}``
    """

    # ── app branch ────────────────────────────────────────────────────
    # Mirror the exact column order produced by
    # ApplicationFeatureExtractor._build_features (minus index + label).
    app_names: list[str] = []

    for col in _LOG_COLS:
        app_names.append(f"{col}_log1p")
    for col in _UNIT_COLS:
        app_names.append(f"ratio_{col}")
    app_names.append("affordable_housing_ratio")
    app_names.extend([
        "application_month_sin",
        "application_month_cos",
        "application_year",
    ])
    for col in _CATEGORICAL_COLS:
        for cat in app_extractor._categories.get(col, []):
            app_names.append(f"{col}_{cat}")

    # ── council branch ────────────────────────────────────────────────
    council_names = list(_COUNCIL_FEATURE_COLS)

    # ── text branch ───────────────────────────────────────────────────
    dim = text_embedder.embedding_dim
    text_names = [f"text_embed_{i}" for i in range(dim)]

    return {"app": app_names, "council": council_names, "text": text_names}
