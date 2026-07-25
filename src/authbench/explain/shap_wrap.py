"""Per-event feature attribution (US-130, `authbench[explain]` extra).

Contribution sources differ by model family, exactly as the spec requires:
tree-based models (IForest, ECOD, HBOS, PCA-adjacent) get SHAP values;
autoencoders get per-dimension reconstruction error (the most informative
proxy for "what looked wrong" without a differentiable attribution method);
the rule engine's own rank-normalized weighted contributions are used
directly — no need for a post-hoc method when the model already IS additive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureContribution:
    feature_name: str
    value: float
    typical_value: float
    contribution: float


def top_contributions(
    feature_names: list[str],
    values: np.ndarray,
    typical_values: np.ndarray,
    contributions: np.ndarray,
    n_top: int = 3,
) -> list[FeatureContribution]:
    """Rank features by |contribution| and return the top `n_top`."""
    order = np.argsort(-np.abs(contributions))[:n_top]
    return [
        FeatureContribution(
            feature_name=feature_names[i],
            value=float(values[i]),
            typical_value=float(typical_values[i]),
            contribution=float(contributions[i]),
        )
        for i in order
    ]


def shap_contributions_tree(
    model: object, x_row: np.ndarray, feature_names: list[str]
) -> np.ndarray:
    """SHAP contributions for a single row, for tree-based models."""
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_row.reshape(1, -1))
    return np.asarray(values).reshape(-1)


def reconstruction_error_contributions(
    x_row: np.ndarray, x_reconstructed_row: np.ndarray
) -> np.ndarray:
    """Per-dimension squared reconstruction error — the autoencoder-family
    proxy for feature contribution (spec section 7, US-130).
    """
    return np.asarray((x_row - x_reconstructed_row) ** 2)
