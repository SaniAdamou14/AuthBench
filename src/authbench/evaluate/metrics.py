"""Core, rank-only metrics (US-125).

Every function here depends only on the *rank* of the scores, never their
absolute scale — matching the `AnomalyScorer` contract. ROC-AUC is computed
and exposed, but always paired with a warning: at LANL's ~1e-7 positive
rate it stays high even for operationally useless models (spec section 6.3).
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROC_AUC_WARNING = (
    "ROC-AUC is reported for literature comparability only. At this dataset's "
    "positive rate, ROC-AUC stays high even for operationally useless models "
    "— see docs/limitations.md. Do not use it to rank models."
)


def auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision (area under the precision-recall curve)."""
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray, *, warn: bool = True) -> float:
    """ROC-AUC, never without its caveat.

    `warn=False` is for callers that scan a whole model catalog and surface
    `ROC_AUC_WARNING` themselves, once — repeating it per model buries the
    message it exists to deliver. The caveat still has to appear somewhere:
    the point of it is that at this positive rate ROC-AUC stays high for
    models no SOC could use, which is exactly what the demo's own table shows.
    """
    if warn:
        warnings.warn(ROC_AUC_WARNING, stacklevel=2)
    return float(roc_auc_score(y_true, scores))


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fraction of true positives among the k highest-scored events."""
    if k <= 0:
        return 0.0
    top_k_idx = np.argsort(-scores)[:k]
    return float(np.mean(y_true[top_k_idx])) if len(top_k_idx) else 0.0


def recall_at_fixed_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Recall at the largest threshold whose false-positive rate does not
    exceed `target_fpr` — for comparability with published ROC-based results.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def degenerate_case_checks() -> dict[str, bool]:
    """US-125: each metric validated against an analytically-known degenerate
    case (all positive, all negative, perfect ranking, inverted ranking).
    Returns a dict of check-name -> passed, used by the test suite.
    """
    results: dict[str, bool] = {}

    y_perfect = np.array([0, 0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    results["perfect_ranking_auc_pr_is_one"] = bool(np.isclose(auc_pr(y_perfect, s_perfect), 1.0))

    s_inverted = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    results["inverted_ranking_precision_at_2_is_zero"] = bool(
        np.isclose(precision_at_k(y_perfect, s_inverted, 2), 0.0)
    )

    y_all_pos = np.ones(5, dtype=int)
    results["all_positive_precision_at_k_is_one"] = bool(
        np.isclose(precision_at_k(y_all_pos, np.random.default_rng(0).random(5), 3), 1.0)
    )

    return results
