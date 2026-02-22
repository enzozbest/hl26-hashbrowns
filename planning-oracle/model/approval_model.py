"""Multi-branch approval neural network with focal loss.

Architecture overview
─────────────────────
Three independent sub-networks (text, application, council) each produce a
128-d representation.  These are concatenated and passed through a fusion
head that outputs a single raw logit.  Applying ``torch.sigmoid`` externally
converts the logit to an approval probability.

The :class:`FocalLoss` criterion down-weights well-classified examples so
the model focuses on hard / borderline decisions — essential when one class
dominates.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Model ────────────────────────────────────────────────────────────────────


class ApprovalModel(nn.Module):
    """Multi-branch feed-forward network for binary approval prediction.

    Each input branch (text, application, council) is processed by its own
    sub-network before the outputs are concatenated and fed into a shared
    fusion head.

    Parameters:
        text_embed_dim: Dimensionality of the pre-computed text embeddings
            (default 384 for ``all-MiniLM-L6-v2``).
        num_app_features: Number of structured application features.
        num_council_features: Number of council-level context features.
    """

    def __init__(
        self,
        text_embed_dim: int = 384,
        num_app_features: int = 20,
        num_council_features: int = 8,
    ) -> None:
        super().__init__()

        # ── text branch ─────────────────────────────────────────────────
        self.text_branch = nn.Sequential(
            nn.Linear(text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # ── application branch ──────────────────────────────────────────
        self.app_branch = nn.Sequential(
            nn.Linear(num_app_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # ── council branch ──────────────────────────────────────────────
        self.council_branch = nn.Sequential(
            nn.Linear(num_council_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        # ── fusion head ─────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning a raw logit.

        Args:
            text_embedding: ``(B, text_embed_dim)``
            app_features: ``(B, num_app_features)``
            council_features: ``(B, num_council_features)``

        Returns:
            Tensor of shape ``(B, 1)`` — raw logit (apply sigmoid for
            probability).
        """
        text_out = self.text_branch(text_embedding)
        app_out = self.app_branch(app_features)
        council_out = self.council_branch(council_features)

        fused = torch.cat([text_out, app_out, council_out], dim=-1)
        return self.fusion(fused)

    @torch.no_grad()
    def predict_proba(
        self,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_features: torch.Tensor,
    ) -> float:
        """Return the approval probability for a single sample.

        Convenience wrapper that applies sigmoid and returns a plain float.
        The model is temporarily set to eval mode and restored afterwards.

        Args:
            text_embedding: ``(1, text_embed_dim)`` or ``(text_embed_dim,)``
            app_features: ``(1, num_app_features)`` or ``(num_app_features,)``
            council_features: ``(1, num_council_features)`` or
                ``(num_council_features,)``

        Returns:
            Scalar probability in ``[0, 1]``.
        """
        was_training = self.training
        self.eval()
        try:
            # Ensure batch dimension
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)
            if app_features.dim() == 1:
                app_features = app_features.unsqueeze(0)
            if council_features.dim() == 1:
                council_features = council_features.unsqueeze(0)

            logit = self.forward(text_embedding, app_features, council_features)
            return torch.sigmoid(logit).squeeze().item()
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def get_branch_outputs(
        self,
        text_embedding: torch.Tensor,
        app_features: torch.Tensor,
        council_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return intermediate branch representations.

        Useful for debugging, visualisation, and attribution.

        Args:
            text_embedding: ``(B, text_embed_dim)``
            app_features: ``(B, num_app_features)``
            council_features: ``(B, num_council_features)``

        Returns:
            Dict with keys ``text``, ``app``, ``council`` mapping to
            ``(B, 128)`` tensors, plus ``fused`` for the concatenation
            and ``logit`` for the final output.
        """
        was_training = self.training
        self.eval()
        try:
            text_out = self.text_branch(text_embedding)
            app_out = self.app_branch(app_features)
            council_out = self.council_branch(council_features)
            fused = torch.cat([text_out, app_out, council_out], dim=-1)
            logit = self.fusion(fused)
            return {
                "text": text_out,
                "app": app_out,
                "council": council_out,
                "fused": fused,
                "logit": logit,
            }
        finally:
            if was_training:
                self.train()


# ── Focal Loss ───────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal loss for imbalanced binary classification.

    .. math::

        FL(p_t) = -\\alpha_t \\,(1 - p_t)^\\gamma \\,\\log(p_t)

    where :math:`p_t = p` when :math:`y = 1` and :math:`1 - p` otherwise,
    and :math:`\\alpha_t` mirrors the same convention.

    Parameters:
        alpha: Weighting factor for the positive class. The negative class
            receives weight ``1 - alpha``.
        gamma: Focusing parameter — higher values down-weight easy examples
            more aggressively.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the mean focal loss over a batch.

        Args:
            logits: Raw logits of shape ``(B,)`` or ``(B, 1)``.
            targets: Binary labels of shape ``(B,)`` or ``(B, 1)``,
                values in ``{0, 1}``.

        Returns:
            Scalar loss tensor.
        """
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        # Numerically stable sigmoid + BCE components
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
        )

        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        return (focal_weight * bce).mean()


# ── Utilities ────────────────────────────────────────────────────────────────


def count_parameters(model: nn.Module) -> int:
    """Log a per-layer parameter summary and return the total count.

    Only trainable parameters are counted.  Each layer's name, shape,
    and count are printed via the module logger at INFO level.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Total number of trainable parameters.
    """
    total = 0
    logger.info("%-50s %15s %12s", "Layer", "Shape", "Params")
    logger.info("-" * 79)
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total += n
            logger.info(
                "%-50s %15s %12s",
                name,
                str(list(param.shape)),
                f"{n:,}",
            )
    logger.info("-" * 79)
    logger.info("%-50s %15s %12s", "TOTAL", "", f"{total:,}")
    return total


# ── Factory / persistence helpers ────────────────────────────────────────────


def build_approval_model(
    text_embed_dim: int = 384,
    num_app_features: int = 20,
    num_council_features: int = 8,
    settings: Optional[Settings] = None,
) -> ApprovalModel:
    """Construct an :class:`ApprovalModel` with settings-driven defaults.

    Args:
        text_embed_dim: Text embedding dimensionality.
        num_app_features: Number of application features.
        num_council_features: Number of council features.
        settings: Application settings (unused currently; reserved for
            future hyperparameter overrides).

    Returns:
        An uninitialised :class:`ApprovalModel`.
    """
    return ApprovalModel(
        text_embed_dim=text_embed_dim,
        num_app_features=num_app_features,
        num_council_features=num_council_features,
    )


def save_model(model: ApprovalModel, path: str) -> None:
    """Save model weights to disk.

    Args:
        model: Trained model instance.
        path: File path for the saved state dict.
    """
    torch.save(model.state_dict(), path)
    logger.info("Model saved to %s", path)


def load_model(
    path: str,
    text_embed_dim: int = 384,
    num_app_features: int = 20,
    num_council_features: int = 8,
) -> ApprovalModel:
    """Load model weights from disk.

    Args:
        path: File path to the saved state dict.
        text_embed_dim: Text embedding dimensionality.
        num_app_features: Number of application features.
        num_council_features: Number of council features.

    Returns:
        An :class:`ApprovalModel` with loaded weights in eval mode.
    """
    model = ApprovalModel(
        text_embed_dim=text_embed_dim,
        num_app_features=num_app_features,
        num_council_features=num_council_features,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    # Support both raw state_dicts and full checkpoint dicts
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded from %s", path)
    return model
