"""Evaluation pipeline: metrics, plots, and segment-level analysis.

Provides a complete post-training evaluation workflow:

* :func:`evaluate_model` — run the model on the test set, compute metrics
  (AUROC, AP, F1, ECE, confusion matrix), and generate publication-quality
  plots (reliability diagram, ROC, PR curve, score distributions).
* :func:`evaluate_by_segment` — compute per-subgroup AUROC and accuracy
  (e.g. by project type, house-count bucket, council region).
* Individual plot functions are separate and reusable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers / CI
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from model.approval_model import ApprovalModel
from model.calibration import TemperatureScaler
from training.train import compute_ece

logger = logging.getLogger(__name__)


# ── Collect predictions ──────────────────────────────────────────────────────


@torch.no_grad()
def _collect_predictions(
    model: ApprovalModel,
    loader: DataLoader,
    scaler: Optional[TemperatureScaler],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run model on *loader* and return raw logits, calibrated probs, labels."""
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        app = batch["app_features"].to(device)
        council = batch["council_features"].to(device)
        text = batch["text_embedding"].to(device)
        labels = batch["label"]

        logits = model(text, app, council).squeeze(-1).cpu()
        all_logits.append(logits.numpy())
        all_labels.append(labels.numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)

    # Uncalibrated probabilities
    raw_probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid

    # Calibrated probabilities
    if scaler is not None:
        cal_probs = scaler.calibrate(
            torch.as_tensor(logits_np),
        ).numpy()
    else:
        cal_probs = raw_probs.copy()

    return {
        "logits": logits_np,
        "raw_probs": raw_probs,
        "cal_probs": cal_probs,
        "labels": labels_np,
    }


# ── Plot functions ───────────────────────────────────────────────────────────


def plot_reliability_diagram(
    labels: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    *,
    n_bins: int = 15,
    save_path: Optional[Path] = None,
) -> None:
    """Reliability diagram with before / after calibration overlaid.

    Args:
        labels: Binary ground-truth, shape ``(N,)``.
        raw_probs: Uncalibrated probabilities.
        cal_probs: Calibrated probabilities.
        n_bins: Number of equal-width bins.
        save_path: If provided, save figure to this path.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _bin_accuracy(probs: np.ndarray) -> np.ndarray:
        accs = np.full(n_bins, np.nan)
        for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            mask = (probs > lo) & (probs <= hi)
            if mask.sum() > 0:
                accs[i] = labels[mask].mean()
        return accs

    raw_acc = _bin_accuracy(raw_probs)
    cal_acc = _bin_accuracy(cal_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")

    valid_raw = ~np.isnan(raw_acc)
    valid_cal = ~np.isnan(cal_acc)
    ax.plot(
        bin_centres[valid_raw], raw_acc[valid_raw],
        "o-", color="tab:red", label="Before calibration",
    )
    ax.plot(
        bin_centres[valid_cal], cal_acc[valid_cal],
        "s-", color="tab:blue", label="After calibration",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reliability diagram saved to %s", save_path)
    plt.close(fig)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    save_path: Optional[Path] = None,
) -> None:
    """ROC curve with AUC annotated.

    Args:
        labels: Binary ground-truth.
        probs: Predicted probabilities.
        save_path: If provided, save figure to this path.
    """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)
    plt.close(fig)


def plot_precision_recall_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    save_path: Optional[Path] = None,
) -> None:
    """Precision-recall curve.

    Args:
        labels: Binary ground-truth.
        probs: Predicted probabilities.
        save_path: If provided, save figure to this path.
    """
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recall, precision, color="tab:green", linewidth=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("PR curve saved to %s", save_path)
    plt.close(fig)


def plot_score_distributions(
    labels: np.ndarray,
    probs: np.ndarray,
    *,
    save_path: Optional[Path] = None,
) -> None:
    """Histogram of predicted probabilities for approved vs refused.

    Args:
        labels: Binary ground-truth.
        probs: Predicted probabilities.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 31)

    approved_mask = labels.astype(int) == 1
    ax.hist(
        probs[approved_mask], bins=bins, alpha=0.6,
        color="tab:green", label="Approved", density=True,
    )
    ax.hist(
        probs[~approved_mask], bins=bins, alpha=0.6,
        color="tab:red", label="Refused", density=True,
    )

    ax.set_xlabel("Predicted probability of approval")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution: Approved vs Refused")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Score distribution saved to %s", save_path)
    plt.close(fig)


# ── Core evaluation ──────────────────────────────────────────────────────────


def _find_optimal_threshold(
    labels: np.ndarray, probs: np.ndarray,
) -> tuple[float, float]:
    """Sweep thresholds to find the one maximising F1.

    Returns:
        ``(best_threshold, best_f1)``
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0.0
    best_t = 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_t, best_f1


def evaluate_model(
    model: ApprovalModel,
    scaler: Optional[TemperatureScaler],
    test_loader: DataLoader,
    feature_names: Optional[dict[str, list[str]]] = None,
    *,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """End-to-end test-set evaluation with metrics and plots.

    Args:
        model: Trained :class:`ApprovalModel` (will be set to eval mode).
        scaler: Fitted :class:`TemperatureScaler` (or *None* to skip
            calibration).
        test_loader: Test DataLoader.
        feature_names: Optional feature-name dict (for logging context).
        output_dir: Directory for saving plots.  If *None*, plots are
            generated but not saved.
        device: Device for inference (inferred from model if *None*).

    Returns:
        Dict containing all computed metrics:

        * ``auroc``, ``ap``, ``f1``, ``optimal_threshold``
        * ``ece_raw``, ``ece_calibrated``
        * ``accuracy``
        * ``confusion_matrix`` — 2x2 list
    """
    if device is None:
        device = next(model.parameters()).device

    # ── collect predictions ──────────────────────────────────────────
    preds = _collect_predictions(model, test_loader, scaler, device)
    labels = preds["labels"]
    raw_probs = preds["raw_probs"]
    cal_probs = preds["cal_probs"]

    # ── scalar metrics ───────────────────────────────────────────────
    unique = np.unique(labels.astype(int))
    if len(unique) < 2:
        auroc = 0.0
        ap = 0.0
        logger.warning("Only one class present in test set — AUROC/AP set to 0")
    else:
        auroc = float(roc_auc_score(labels, cal_probs))
        ap = float(average_precision_score(labels, cal_probs))

    opt_thresh, best_f1 = _find_optimal_threshold(labels, cal_probs)

    hard_preds_05 = (cal_probs >= 0.5).astype(int)
    accuracy = float((hard_preds_05 == labels.astype(int)).mean())

    ece_raw = compute_ece(raw_probs, labels)
    ece_cal = compute_ece(cal_probs, labels)

    cm = confusion_matrix(
        labels.astype(int), hard_preds_05, labels=[0, 1],
    ).tolist()

    metrics = {
        "auroc": auroc,
        "ap": ap,
        "f1": best_f1,
        "optimal_threshold": opt_thresh,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }

    logger.info(
        "Evaluation: AUROC=%.4f  AP=%.4f  F1=%.4f (t=%.2f)  "
        "ECE_raw=%.4f  ECE_cal=%.4f  Acc=%.4f",
        auroc, ap, best_f1, opt_thresh, ece_raw, ece_cal, accuracy,
    )
    logger.info("Confusion matrix (TN, FP / FN, TP): %s", cm)

    # ── plots ────────────────────────────────────────────────────────
    save = output_dir is not None
    od = Path(output_dir) if output_dir else Path("")

    plot_reliability_diagram(
        labels, raw_probs, cal_probs,
        save_path=od / "reliability_diagram.png" if save else None,
    )

    if len(unique) >= 2:
        plot_roc_curve(
            labels, cal_probs,
            save_path=od / "roc_curve.png" if save else None,
        )
        plot_precision_recall_curve(
            labels, cal_probs,
            save_path=od / "pr_curve.png" if save else None,
        )

    plot_score_distributions(
        labels, cal_probs,
        save_path=od / "score_distributions.png" if save else None,
    )

    return metrics


# ── Segment-level evaluation ─────────────────────────────────────────────────


def evaluate_by_segment(
    model: ApprovalModel,
    scaler: Optional[TemperatureScaler],
    test_loader: DataLoader,
    segments: dict[str, np.ndarray],
    *,
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, float]]:
    """Compute per-segment AUROC and accuracy.

    Args:
        model: Trained :class:`ApprovalModel`.
        scaler: Fitted :class:`TemperatureScaler` (or *None*).
        test_loader: Test DataLoader.
        segments: Mapping from segment name to a boolean mask of shape
            ``(N_test,)`` indicating which samples belong to that segment.
        device: Inference device.

    Returns:
        ``{segment_name: {"auroc": float, "accuracy": float, "n": int}}``
        for each segment that has at least two classes represented.
    """
    if device is None:
        device = next(model.parameters()).device

    preds = _collect_predictions(model, test_loader, scaler, device)
    labels = preds["labels"]
    probs = preds["cal_probs"]

    results: dict[str, dict[str, float]] = {}

    for name, mask in segments.items():
        mask = mask.astype(bool)
        n = int(mask.sum())
        if n == 0:
            logger.warning("Segment '%s' is empty — skipping", name)
            continue

        seg_labels = labels[mask]
        seg_probs = probs[mask]

        hard = (seg_probs >= 0.5).astype(int)
        acc = float((hard == seg_labels.astype(int)).mean())

        unique = np.unique(seg_labels.astype(int))
        if len(unique) < 2:
            auroc = float("nan")
            logger.warning(
                "Segment '%s' has only one class — AUROC undefined", name,
            )
        else:
            auroc = float(roc_auc_score(seg_labels, seg_probs))

        results[name] = {"auroc": auroc, "accuracy": acc, "n": n}
        logger.info(
            "Segment %-30s  n=%5d  AUROC=%.4f  Acc=%.4f",
            name, n,
            auroc if not np.isnan(auroc) else -1.0,
            acc,
        )

    return results
