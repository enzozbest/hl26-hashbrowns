"""Post-hoc probability calibration via temperature scaling.

After training, a neural network's predicted probabilities are often
*miscalibrated* — a prediction of "80 % approved" may actually correspond
to a 65 % empirical rate.  Temperature scaling learns a single scalar
*T* > 0 that stretches or compresses the logit distribution so that
``sigmoid(logit / T)`` produces well-calibrated probabilities.

The :class:`TemperatureScaler` is fitted on the **validation set** (never
the training set) using LBFGS to minimise the negative log-likelihood.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.approval_model import ApprovalModel

logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaler for binary logits.

    After :meth:`fit`, call :meth:`calibrate` to convert raw logits into
    calibrated probabilities.

    Attributes:
        temperature: Learned temperature parameter (initialised to 1.5).
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

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

    def fit(
        self,
        model: ApprovalModel,
        val_loader: DataLoader,
        *,
        lr: float = 0.01,
        max_iter: int = 50,
        device: Optional[torch.device] = None,
    ) -> float:
        """Optimise temperature *T* on the validation set.

        Uses LBFGS to minimise the binary cross-entropy (NLL) of
        ``sigmoid(logit / T)`` against the true labels.

        Args:
            model: Trained (frozen) :class:`ApprovalModel`.
            val_loader: Validation DataLoader.
            lr: LBFGS learning rate.
            max_iter: LBFGS maximum iterations.
            device: Device for optimisation (inferred from model if *None*).

        Returns:
            Final NLL loss after optimisation.
        """
        if device is None:
            device = next(model.parameters()).device

        logits, labels = self._collect_logits(model, val_loader, device)
        self.to(device)

        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter,
        )

        final_loss = torch.tensor(0.0)

        def _closure() -> torch.Tensor:
            nonlocal final_loss
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            final_loss = loss.detach()
            return loss

        optimizer.step(_closure)

        logger.info(
            "Temperature scaling fitted: T=%.4f  NLL=%.4f",
            self.temperature.item(),
            final_loss.item(),
        )
        return final_loss.item()

    # ── inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw logits — any shape (scalar, 1-D, or 2-D).

        Returns:
            ``sigmoid(logits / T)`` with the same shape as input.
        """
        return torch.sigmoid(logits / self.temperature)

    # ── persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save the fitted temperature to disk.

        Args:
            path: File path (pickle).
        """
        with open(path, "wb") as f:
            pickle.dump({"temperature": self.temperature.item()}, f)
        logger.info("TemperatureScaler saved to %s (T=%.4f)", path, self.temperature.item())

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        """Load a previously fitted scaler.

        Args:
            path: File path produced by :meth:`save`.

        Returns:
            A :class:`TemperatureScaler` with the restored temperature.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)  # noqa: S301
        inst = cls()
        inst.temperature = nn.Parameter(
            torch.tensor([state["temperature"]]),
        )
        logger.info("TemperatureScaler loaded from %s (T=%.4f)", path, inst.temperature.item())
        return inst
