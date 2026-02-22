"""Stage 3: SHAP-based feature attribution.

Explains individual predictions by computing SHAP values, identifying which
features contributed most to the approval probability.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from neural_network.model.approval_model import ApprovalModel


class SHAPExplainer:
    """Wrapper around SHAP's DeepExplainer for the approval model.

    Parameters:
        model: A trained ``ApprovalNet`` instance.
        background_data: A sample of training data for SHAP background
            (typically 100–500 rows).
    """

    def __init__(
        self,
        model: ApprovalModel,
        background_data: np.ndarray,
    ) -> None:
        self._model = model
        self._background = background_data
        self._explainer = None  # Lazy-initialised

    def _init_explainer(self) -> None:
        """Initialise the SHAP DeepExplainer.

        Called automatically on the first explain call.
        """
        raise NotImplementedError

    def explain(self, features: np.ndarray) -> np.ndarray:
        """Compute SHAP values for one or more inputs.

        Args:
            features: Input array of shape ``(n, input_dim)``.

        Returns:
            SHAP values array of shape ``(n, input_dim)``.
        """
        raise NotImplementedError

    def top_features(
        self,
        features: np.ndarray,
        feature_names: list[str],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k most influential features for a single prediction.

        Args:
            features: Input array of shape ``(1, input_dim)``.
            feature_names: Human-readable names for each feature dimension.
            top_k: Number of top features to return.

        Returns:
            List of dicts with ``feature``, ``shap_value``, and ``direction``
            (positive / negative) keys, sorted by absolute impact.
        """
        raise NotImplementedError

    def summary_plot(
        self,
        features: np.ndarray,
        feature_names: list[str],
        *,
        save_path: Optional[str] = None,
    ) -> None:
        """Generate a SHAP summary plot.

        Args:
            features: Input array of shape ``(n, input_dim)``.
            feature_names: Human-readable feature names.
            save_path: If provided, save the plot to this file path instead
                of displaying it.
        """
        raise NotImplementedError
