"""Post-hoc probability calibration: Platt scaling + isotonic regression.

After training, a neural network's predicted probabilities are often
*miscalibrated* — a prediction of "80 % approved" may actually correspond
to a 65 % empirical rate.  This module provides a three-phase calibration
pipeline:

1. **Platt scaling** (temperature + bias) — LBFGS minimises NLL on the
   validation set.  Corrects the *linear* component of miscalibration.
2. **ECE fine-tuning** — Adam minimises a differentiable ECE proxy to
   push calibration error down further within the linear constraint.
3. **Isotonic regression** — a non-parametric monotonic mapping that
   corrects any *residual non-linear* miscalibration (e.g. from focal
   loss or weighted sampling).

The :class:`TemperatureScaler` is fitted on the **validation set** (never
the training set).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader

from neural_network.model.approval_model import ApprovalModel

logger = logging.getLogger(__name__)


# ── Differentiable ECE proxy ─────────────────────────────────────────────────


def _differentiable_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    sharpness: float = 30.0,
) -> torch.Tensor:
    """Soft-binned ECE that is differentiable w.r.t. *probs*.

    Uses sigmoid-based soft bin membership instead of hard thresholding so
    gradients flow through the bin assignment.

    Args:
        probs: Predicted probabilities, shape ``(N,)``.
        labels: Binary ground-truth, shape ``(N,)``.
        n_bins: Number of equal-width bins.
        sharpness: Controls how sharply the soft bins approximate hard bins.
            Higher = closer to hard ECE but noisier gradients.
    """
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device)
    ece = torch.tensor(0.0, device=probs.device)
    n = probs.shape[0]

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Soft membership: product of two sigmoids gives a smooth indicator
        weight = (
            torch.sigmoid(sharpness * (probs - lo))
            * torch.sigmoid(sharpness * (hi - probs))
        )
        count = weight.sum()
        if count < 1.0:
            continue
        avg_conf = (weight * probs).sum() / count
        avg_acc = (weight * labels).sum() / count
        ece = ece + (count / n) * torch.abs(avg_acc - avg_conf)

    return ece


def _hard_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Standard (non-differentiable) ECE for logging."""
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
        ece += (count / n) * abs(probs[mask].mean() - labels[mask].mean())
    return float(ece)


class TemperatureScaler(nn.Module):
    """Platt scaler + isotonic regression for binary logits.

    Three-phase calibration:

    1. Platt scaling: ``P = sigmoid((logit + bias) / temperature)``
    2. ECE fine-tuning: gradient descent on a differentiable ECE proxy
    3. Isotonic regression: non-parametric monotonic correction for any
       residual non-linear miscalibration (critical when using focal loss)

    After :meth:`fit`, call :meth:`calibrate` to convert raw logits into
    calibrated probabilities.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.bias = nn.Parameter(torch.zeros(1))
        self._isotonic: Optional[IsotonicRegression] = None

    # ── fitting ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_logits(
        self,
        model: ApprovalModel,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run *model* on every batch in *loader* and collect logits + labels."""
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in loader:
            app = batch["app_features"].to(device)
            council = batch["council_features"].to(device)
            text = batch["text_embedding"].to(device)
            labels = batch["label"].to(device)

            logits = model(text, app, council).squeeze(-1)
            all_logits.append(logits)
            all_labels.append(labels)

        return torch.cat(all_logits), torch.cat(all_labels)

    def _scaled_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling: ``(logits + bias) / temperature``."""
        return (logits + self.bias) / self.temperature

    def fit(
        self,
        model: ApprovalModel,
        val_loader: DataLoader,
        *,
        lr: float = 0.01,
        max_iter: int = 200,
        ece_lr: float = 0.001,
        ece_steps: int = 500,
        target_ece: float = 0.03,
        device: Optional[torch.device] = None,
    ) -> float:
        """Optimise calibration parameters on the validation set.

        **Phase 1 (NLL):** LBFGS minimises the binary cross-entropy of
        ``sigmoid((logit + b) / T)`` against the true labels.

        **Phase 2 (ECE):** If ECE is still above *target_ece*, Adam
        fine-tunes *T* and *b* using a differentiable ECE proxy.

        **Phase 3 (Isotonic):** If ECE is *still* above *target_ece*,
        fit an isotonic regression on the Platt-scaled probabilities to
        correct any residual non-linear miscalibration (e.g. from focal
        loss or weighted sampling).

        Args:
            model: Trained (frozen) :class:`ApprovalModel`.
            val_loader: Validation DataLoader.
            lr: LBFGS learning rate for Phase 1.
            max_iter: Max LBFGS iterations per outer step.
            ece_lr: Adam learning rate for Phase 2.
            ece_steps: Max gradient steps for Phase 2.
            target_ece: Stop early when ECE drops below this.
            device: Device for optimisation (inferred from model if *None*).

        Returns:
            Learned temperature value.
        """
        if device is None:
            device = next(model.parameters()).device

        logits, labels = self._collect_logits(model, val_loader, device)
        self.to(device)

        labels_np = labels.cpu().numpy()

        # ── Phase 1: NLL optimisation via LBFGS ──────────────────────
        optimizer = torch.optim.LBFGS(
            [self.temperature, self.bias],
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
        )

        prev_loss = float("inf")
        for step in range(20):
            def _closure() -> torch.Tensor:
                optimizer.zero_grad()
                scaled = self._scaled_logits(logits)
                loss = F.binary_cross_entropy_with_logits(scaled, labels)
                loss.backward()
                return loss

            loss = optimizer.step(_closure)
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss
            if abs(prev_loss - loss_val) < 1e-10:
                logger.info("  NLL converged at outer step %d", step + 1)
                break
            prev_loss = loss_val

        with torch.no_grad():
            probs_np = torch.sigmoid(self._scaled_logits(logits)).cpu().numpy()
        ece_after_nll = _hard_ece(probs_np, labels_np)

        logger.info(
            "Phase 1 (NLL): T=%.4f  b=%.4f  NLL=%.4f  ECE=%.4f",
            self.temperature.item(),
            self.bias.item(),
            prev_loss,
            ece_after_nll,
        )

        # ── Phase 2: ECE fine-tuning via Adam ────────────────────────
        current_ece = ece_after_nll

        if current_ece > target_ece:
            logger.info(
                "Phase 2 (ECE): fine-tuning to push ECE below %.4f …",
                target_ece,
            )

            ece_optimizer = torch.optim.Adam(
                [self.temperature, self.bias], lr=ece_lr,
            )
            ece_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                ece_optimizer, mode="min", factor=0.5, patience=50,
            )

            best_ece = current_ece
            best_state = {
                "temperature": self.temperature.data.clone(),
                "bias": self.bias.data.clone(),
            }

            for step in range(1, ece_steps + 1):
                ece_optimizer.zero_grad()
                scaled = self._scaled_logits(logits)
                probs = torch.sigmoid(scaled)

                nll = F.binary_cross_entropy_with_logits(scaled, labels)
                ece_loss = _differentiable_ece(probs, labels)
                loss = 0.1 * nll + ece_loss
                loss.backward()
                ece_optimizer.step()
                ece_scheduler.step(ece_loss)

                with torch.no_grad():
                    self.temperature.data.clamp_(min=0.01)

                if step % 50 == 0 or step == ece_steps:
                    with torch.no_grad():
                        cur_probs = torch.sigmoid(
                            self._scaled_logits(logits),
                        ).cpu().numpy()
                    cur_ece = _hard_ece(cur_probs, labels_np)

                    if cur_ece < best_ece:
                        best_ece = cur_ece
                        best_state = {
                            "temperature": self.temperature.data.clone(),
                            "bias": self.bias.data.clone(),
                        }

                    logger.info(
                        "  step %d/%d: ECE=%.4f  best=%.4f  T=%.4f  b=%.4f",
                        step, ece_steps, cur_ece, best_ece,
                        self.temperature.item(), self.bias.item(),
                    )

                    if best_ece <= target_ece:
                        logger.info(
                            "  ECE target reached (%.4f <= %.4f)",
                            best_ece, target_ece,
                        )
                        break

            with torch.no_grad():
                self.temperature.data.copy_(best_state["temperature"])
                self.bias.data.copy_(best_state["bias"])

            current_ece = best_ece
            logger.info(
                "Phase 2 done: T=%.4f  b=%.4f  ECE=%.4f → %.4f",
                self.temperature.item(),
                self.bias.item(),
                ece_after_nll,
                current_ece,
            )
        else:
            logger.info(
                "ECE already at %.4f (below target %.4f) — skipping Phase 2",
                current_ece, target_ece,
            )

        # ── Phase 3: Isotonic regression (non-parametric) ────────────
        if current_ece > target_ece:
            logger.info(
                "Phase 3 (Isotonic): fitting non-parametric correction "
                "(ECE=%.4f still above %.4f) …",
                current_ece,
                target_ece,
            )

            # Get Platt-scaled probabilities as input to isotonic regression
            with torch.no_grad():
                platt_probs = (
                    torch.sigmoid(self._scaled_logits(logits)).cpu().numpy()
                )

            self._isotonic = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip",
            )
            self._isotonic.fit(platt_probs, labels_np)

            iso_probs = self._isotonic.predict(platt_probs)
            ece_after_iso = _hard_ece(iso_probs, labels_np)

            logger.info(
                "Phase 3 done: ECE=%.4f → %.4f",
                current_ece,
                ece_after_iso,
            )
            current_ece = ece_after_iso
        else:
            logger.info(
                "ECE at %.4f (below target %.4f) — skipping Phase 3",
                current_ece, target_ece,
            )

        logger.info(
            "Calibration complete: T=%.4f  b=%.4f  isotonic=%s  final_ECE=%.4f",
            self.temperature.item(),
            self.bias.item(),
            self._isotonic is not None,
            current_ece,
        )
        return self.temperature.item()

    # ── inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply full calibration pipeline and return calibrated probabilities.

        1. Platt scaling: ``sigmoid((logits + bias) / T)``
        2. Isotonic regression (if fitted): non-parametric monotonic correction

        Args:
            logits: Raw logits — any shape (scalar, 1-D, or 2-D).

        Returns:
            Calibrated probabilities with the same shape as input.
        """
        probs = torch.sigmoid(self._scaled_logits(logits))

        if self._isotonic is not None:
            shape = probs.shape
            probs_np = probs.cpu().numpy().ravel()
            cal_np = self._isotonic.predict(probs_np)
            probs = torch.as_tensor(
                cal_np.reshape(shape), dtype=probs.dtype,
            )

        return probs

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the fitted parameters to disk.

        Args:
            path: File path (pickle).
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "temperature": self.temperature.item(),
                    "bias": self.bias.item(),
                    "isotonic": self._isotonic,
                },
                f,
            )
        logger.info(
            "TemperatureScaler saved to %s (T=%.4f, b=%.4f, isotonic=%s)",
            path,
            self.temperature.item(),
            self.bias.item(),
            self._isotonic is not None,
        )

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        """Load a previously fitted scaler.

        Args:
            path: File path produced by :meth:`save`.

        Returns:
            A :class:`TemperatureScaler` with the restored parameters.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls()
        inst.temperature = nn.Parameter(
            torch.tensor([state["temperature"]]),
        )
        inst.bias = nn.Parameter(
            torch.tensor([state.get("bias", 0.0)]),
        )
        inst._isotonic = state.get("isotonic", None)
        logger.info(
            "TemperatureScaler loaded from %s (T=%.4f, b=%.4f, isotonic=%s)",
            path,
            inst.temperature.item(),
            inst.bias.item(),
            inst._isotonic is not None,
        )
        return inst
