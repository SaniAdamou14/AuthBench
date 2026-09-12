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
from authbench.evaluate.plots import (
    CAMPAIGN_RECALL_FIGURE,
    _shade_tie_bracket,
    plot_campaign_recall_vs_budget,
    plt,
)

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


def test_the_tie_bracket_is_shaded_only_where_it_has_width() -> None:
    """The shading has to mean something, which means it has to be absent.

    A band drawn under every curve would say nothing; drawn only where the
    tie-break could have changed the answer, its presence marks the soft
    numbers and its absence certifies the rest. A curve carrying no bracket at
    all — everything built before `compute_budget_curve` reported one — must
    also draw nothing rather than crash.
    """
    fig, ax = plt.subplots()
    try:
        soft = _curve("M0b_always_fail", [0.0, 0.0, 0.0, 0.75])
        soft.campaign_recall_min = [0.0, 0.0, 0.0, 0.0]
        soft.campaign_recall_max = [0.0, 0.5, 1.0, 1.0]

        firm = _curve("M1_rules", [0.0, 0.5, 1.0, 1.0])
        firm.campaign_recall_min = list(firm.campaign_recall)
        firm.campaign_recall_max = list(firm.campaign_recall)

        assert _shade_tie_bracket(ax, soft, "0.55") is True
        assert _shade_tie_bracket(ax, firm, "0.55") is False
        assert _shade_tie_bracket(ax, _curve("M2a_pair_rarity", [0.0] * 4), "0.55") is False
    finally:
        plt.close(fig)
