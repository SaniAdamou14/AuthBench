from __future__ import annotations

import numpy as np
import pytest

from authbench.evaluate.metrics import (
    auc_pr,
    degenerate_case_checks,
    precision_at_k,
    recall_at_fixed_fpr,
    roc_auc,
)


def test_degenerate_cases_all_pass() -> None:
    results = degenerate_case_checks()
    for name, passed in results.items():
        assert passed, f"degenerate case failed: {name}"


def test_precision_at_k_perfect_ranking() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    assert precision_at_k(y, scores, 2) == 1.0


def test_precision_at_k_zero_k() -> None:
    y = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    assert precision_at_k(y, scores, 0) == 0.0


def test_roc_auc_emits_warning() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    with pytest.warns(UserWarning, match="ROC-AUC"):
        value = roc_auc(y, scores)
    assert 0.0 <= value <= 1.0


def test_recall_at_fixed_fpr_bounds() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.1).astype(int)
    scores = rng.random(200)
    recall = recall_at_fixed_fpr(y, scores, target_fpr=1e-4)
    assert 0.0 <= recall <= 1.0


def test_auc_pr_perfect_vs_random() -> None:
    y = np.array([0] * 90 + [1] * 10)
    perfect_scores = np.array([0.0] * 90 + [1.0] * 10)
    assert auc_pr(y, perfect_scores) == pytest.approx(1.0)
