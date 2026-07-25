from __future__ import annotations

import numpy as np

from authbench.explain.alert_card import build_alert_card
from authbench.explain.shap_wrap import (
    reconstruction_error_contributions,
    top_contributions,
)


def test_reconstruction_error_contributions_is_squared_diff() -> None:
    x = np.array([1.0, 2.0, 3.0])
    x_hat = np.array([1.0, 0.0, 5.0])
    contributions = reconstruction_error_contributions(x, x_hat)
    assert np.allclose(contributions, [0.0, 4.0, 4.0])


def test_top_contributions_ranks_by_absolute_value() -> None:
    names = ["a", "b", "c", "d"]
    values = np.array([1.0, 2.0, 3.0, 4.0])
    typical = np.array([0.0, 0.0, 0.0, 0.0])
    contributions = np.array([0.1, -5.0, 0.2, 3.0])

    top = top_contributions(names, values, typical, contributions, n_top=2)

    assert [c.feature_name for c in top] == ["b", "d"]


def test_alert_card_renders_markdown_with_mitre_section() -> None:
    top = top_contributions(
        ["pair_is_new"], np.array([1.0]), np.array([0.0]), np.array([2.5]), n_top=1
    )
    card = build_alert_card(
        event_id=42,
        src_user="alice",
        src_computer="C1",
        dst_computer="C2",
        time=12345,
        score=0.87,
        rank_in_day=3,
        top_features=top,
        mitre_techniques=["T1078"],
    )
    markdown = card.render_markdown()
    assert "event 42" in markdown
    assert "pair_is_new" in markdown
    assert "T1078" in markdown
