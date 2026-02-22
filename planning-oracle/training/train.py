"""Training loop for the multi-branch approval model.

Orchestrates the full training pipeline: dataset construction, model
initialisation, optimiser & scheduler setup, epoch iteration with early
stopping, checkpoint management, and final test-set evaluation.

Run directly::

    python -m training.train --epochs 50 --batch-size 256 --lr 1e-3
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from config.settings import Settings, get_settings
from data.api_client import PlanningAPIClient
from features.application import ApplicationFeatureExtractor
from features.council import CouncilFeatureExtractor
from features.text import TextEmbedder
from model.approval_model import ApprovalModel, count_parameters
from training.dataset import PlanningDataset, build_dataloaders, build_datasets

logger = logging.getLogger(__name__)

# ── Calibration metric ───────────────────────────────────────────────────────


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error.

    Partitions predictions into *n_bins* equally-spaced confidence bins,
    computes ``|accuracy - confidence|`` in each, and returns the weighted
    average (weights = bin size / total).

    Args:
        probs: Predicted probabilities, shape ``(N,)``.
        labels: Binary ground-truth labels, shape ``(N,)``.
        n_bins: Number of bins.

    Returns:
        Scalar ECE in ``[0, 1]``.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs > lo) & (probs <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = probs[mask].mean()
        avg_acc = labels[mask].mean()
        ece += (count / n) * abs(avg_acc - avg_conf)

    return float(ece)


# ── Single-epoch helpers ─────────────────────────────────────────────────────


def _train_one_epoch(
    model: ApprovalModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """Run one training epoch and return the mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        app = batch["app_features"].to(device)
        council = batch["council_features"].to(device)
        text = batch["text_embedding"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(text, app, council).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate(
    model: ApprovalModel,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run a validation pass and return loss + classification metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        app = batch["app_features"].to(device)
        council = batch["council_features"].to(device)
        text = batch["text_embedding"].to(device)
        labels = batch["label"].to(device)

        logits = model(text, app, council).squeeze(-1)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)

    preds = (probs_np >= 0.5).astype(int)
    accuracy = (preds == labels_np.astype(int)).mean()

    # Handle edge cases where only one class is present
    unique = np.unique(labels_np.astype(int))
    if len(unique) < 2:
        auroc = 0.0
        ap = 0.0
    else:
        auroc = float(roc_auc_score(labels_np, probs_np))
        ap = float(average_precision_score(labels_np, probs_np))

    ece = compute_ece(probs_np, labels_np)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auroc": auroc,
        "ap": ap,
        "ece": ece,
    }


# ── Checkpoint helpers ───────────────────────────────────────────────────────


def _save_checkpoint(
    path: Path,
    model: ApprovalModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auroc: float,
    text_embed_dim: int,
    num_app_features: int,
    num_council_features: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_auroc": best_val_auroc,
            "text_embed_dim": text_embed_dim,
            "num_app_features": num_app_features,
            "num_council_features": num_council_features,
        },
        path,
    )
    logger.info("Checkpoint saved → %s (epoch %d, val_auroc=%.4f)", path, epoch, best_val_auroc)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


# ── Main training pipeline ───────────────────────────────────────────────────


async def train_model(
    config: Settings,
    client: PlanningAPIClient,
    *,
    council_ids: list[int],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> tuple[ApprovalModel, dict]:
    """Full training pipeline: build data → train → evaluate on test set.

    Steps:

    1. Build datasets and dataloaders (temporal split, feature extraction).
    2. Initialise :class:`ApprovalModel` with correct feature dimensions.
    3. Set up AdamW optimiser, cosine-annealing scheduler, and focal loss.
    4. Train for *config.epochs* with early stopping (patience 10 on
       val AUROC).
    5. Load the best checkpoint and evaluate on the held-out test set.

    Args:
        config: Application settings (learning rate, epochs, batch size,
            checkpoint directory, embedding model, etc.).
        client: Configured :class:`PlanningAPIClient` for fetching data.
        council_ids: List of council identifiers to fetch data for.
        date_from: Optional ISO-8601 start date for the search window.
        date_to: Optional ISO-8601 end date for the search window.

    Returns:
        ``(model, metrics)`` where *model* is the best checkpoint loaded
        in eval mode and *metrics* is a dict of per-epoch training history
        plus final test-set results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", device)

    # ── 1. Build datasets ────────────────────────────────────────────
    text_embedder = TextEmbedder(
        model_name=config.text_encoder_model,
        settings=config,
    )
    app_extractor = ApplicationFeatureExtractor()
    council_extractor = CouncilFeatureExtractor()

    train_ds, val_ds, test_ds = await build_datasets(
        client, text_embedder, app_extractor, council_extractor,
        council_ids=council_ids,
        date_from=date_from,
        date_to=date_to,
        checkpoint_dir=config.checkpoint_dir,
    )
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=config.batch_size,
    )

    # ── 2. Initialise model ──────────────────────────────────────────
    text_embed_dim = train_ds.text_embeddings.shape[1]
    num_app_features = train_ds.app_features.shape[1]
    num_council_features = train_ds.council_features.shape[1]

    model = ApprovalModel(
        text_embed_dim=text_embed_dim,
        num_app_features=num_app_features,
        num_council_features=num_council_features,
    ).to(device)

    total_params = count_parameters(model)
    logger.info("Model has %s trainable parameters", f"{total_params:,}")

    # ── 3. Optimiser, scheduler, loss ────────────────────────────────
    logger.info("Config learning_rate=%.2e", config.learning_rate)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4,
    )

    # Two-phase scheduler:
    #   Phase 1 — LinearLR warmup over the first 3 epochs, ramping from
    #             10% of the target LR up to 100%.  This prevents the
    #             large gradient updates on epoch 1 that cause val AUROC
    #             to peak immediately then decay.
    #   Phase 2 — ReduceLROnPlateau halves the LR after 5 epochs without
    #             improvement, giving the model time to settle after the
    #             warmup before the first reduction.
    warmup_epochs = 3
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,   # begin at 10% of config.learning_rate
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",     # we pass -val_auroc so lower = better
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    # No pos_weight — we want the model calibrated to the true class
    # distribution. The model should learn to predict "approved" most
    # of the time because that reflects reality (~87% of applications
    # are approved). Reweighting distorts probabilities and hurts ECE.
    train_labels = train_ds.labels.numpy()
    n_pos = int(train_labels.sum())
    n_neg = int(len(train_labels) - n_pos)
    logger.info(
        "Class balance — pos=%d  neg=%d  (%.1f%% positive)",
        n_pos, n_neg, 100.0 * n_pos / max(len(train_labels), 1),
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # ── 4. Training loop with early stopping ─────────────────────────
    checkpoint_dir = Path(config.checkpoint_dir)
    best_path = checkpoint_dir / "best_model.pt"

    best_val_auroc = -1.0
    patience = 15  # increased from 10 to give post-warmup LR reductions time
    epochs_without_improvement = 0

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auroc": [],
        "val_ap": [],
        "val_ece": [],
    }

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_metrics = _validate(model, val_loader, criterion, device)

        # Warmup phase: step LinearLR for the first warmup_epochs.
        # Plateau phase: step ReduceLROnPlateau every epoch after warmup.
        if epoch <= warmup_epochs:
            warmup.step()
        else:
            plateau.step(-val_metrics["auroc"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_ap"].append(val_metrics["ap"])
        history["val_ece"].append(val_metrics["ece"])

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f "
            "val_auroc=%.4f val_ap=%.4f val_ece=%.4f lr=%.2e",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["auroc"],
            val_metrics["ap"],
            val_metrics["ece"],
            current_lr,
        )

        # ── checkpoint on improvement ────────────────────────────────
        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            epochs_without_improvement = 0
            _save_checkpoint(
                best_path, model, optimizer, epoch, best_val_auroc,
                text_embed_dim, num_app_features, num_council_features,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs)",
                    epoch, patience,
                )
                break

    # ── 5. Save feature extractors alongside the model ───────────────
    app_extractor.save(checkpoint_dir / "app_extractor.pkl")
    council_extractor.save(checkpoint_dir / "council_extractor.pkl")
    logger.info("Feature extractors saved to %s", checkpoint_dir)

    # ── 6. Load best checkpoint & evaluate on test set ───────────────
    if best_path.exists():
        ckpt = _load_checkpoint(best_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded best checkpoint from epoch %d (val_auroc=%.4f)",
            ckpt["epoch"], ckpt["best_val_auroc"],
        )
    else:
        logger.warning("No checkpoint found — using final model weights")

    test_metrics = _validate(model, test_loader, criterion, device)
    logger.info(
        "Test results: loss=%.4f auroc=%.4f ap=%.4f ece=%.4f accuracy=%.4f",
        test_metrics["loss"],
        test_metrics["auroc"],
        test_metrics["ap"],
        test_metrics["ece"],
        test_metrics["accuracy"],
    )

    metrics = {
        "history": history,
        "best_epoch": (
            _load_checkpoint(best_path, device)["epoch"]
            if best_path.exists()
            else config.epochs
        ),
        "best_val_auroc": best_val_auroc,
        "test": test_metrics,
    }

    model.eval()
    return model, metrics


# ── CLI entrypoint ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train the planning-oracle approval model.",
    )
    parser.add_argument(
        "--config", default=".env",
        help="Path to .env configuration file (default: .env)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size (overrides config)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Checkpoint directory (overrides config)",
    )
    parser.add_argument(
        "--councils", nargs="+", type=int, default=[],
        help="Council IDs (integers) to fetch data for",
    )
    parser.add_argument(
        "--date-from", default=None,
        help="ISO-8601 start date for the search window",
    )
    parser.add_argument(
        "--date-to", default=None,
        help="ISO-8601 end date for the search window",
    )

    args = parser.parse_args(argv)

    # Build settings from the env file, then apply CLI overrides
    config = Settings(_env_file=args.config)
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    async def _run() -> tuple[ApprovalModel, dict]:
        async with PlanningAPIClient(settings=config) as client:
            return await train_model(
                config, client,
                council_ids=args.councils,
                date_from=args.date_from,
                date_to=args.date_to,
            )

    model, metrics = asyncio.run(_run())

    logger.info("Training complete.")
    logger.info("  Best val AUROC: %.4f (epoch %d)", metrics["best_val_auroc"], metrics["best_epoch"])
    logger.info("  Test AUROC:     %.4f", metrics["test"]["auroc"])
    logger.info("  Test AP:        %.4f", metrics["test"]["ap"])
    logger.info("  Test ECE:       %.4f", metrics["test"]["ece"])


if __name__ == "__main__":
    main()