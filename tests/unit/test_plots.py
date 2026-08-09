"""The headline figure is a pipeline output, so it is covered like one.

`plot_campaign_recall_vs_budget` is called at the very end of both
`authbench demo` and the DVC `train_eval` stage — the point where a crash
costs a full run. These tests exercise the paths that would only otherwise
be discovered there: the floor/model split, the tie case the marker scheme
exists for, and the empty-input guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from authbench.evaluate.budget import BudgetCurve
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget

BUDGETS = [10, 50, 100, 500]


def _curve(name: str, campaign_recall: list[float]) -> BudgetCurve:
    return BudgetCurve(
        model_name=name,
        budgets=BUDGETS,
        event_recall=[r / 2 for r in campaign_recall],
        campaign_recall=campaign_recall,
    )


def test_writes_a_non_empty_png_and_creates_missing_parent_directories(tmp_path: Path) -> None:
    out_path = tmp_path / "figures" / CAMPAIGN_RECALL_FIGURE
    curves = [
        _curve("M0a_random", [0.0, 0.0, 0.0, 0.0]),
        _curve("M1_rules", [0.0, 0.5, 1.0, 1.0]),
        _curve("M3a_iforest", [0.0, 0.0, 0.5, 1.0]),
    ]

    written = plot_campaign_recall_vs_budget(curves, out_path, subtitle="demo sample")

    assert written == out_path
    assert out_path.stat().st_size > 0


def test_handles_models_that_tie_at_every_budget(tmp_path: Path) -> None:
    """Campaign recall is a ratio over a handful of campaigns, so exact ties
    are the normal case, not an edge case.
    """
    out_path = tmp_path / CAMPAIGN_RECALL_FIGURE
    tied = [_curve(f"M{i}_model", [1.0, 1.0, 1.0, 1.0]) for i in range(1, 10)]

    assert plot_campaign_recall_vs_budget(tied, out_path).stat().st_size > 0


def test_refuses_to_write_a_figure_with_no_curves(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one BudgetCurve"):
        plot_campaign_recall_vs_budget([], tmp_path / CAMPAIGN_RECALL_FIGURE)
