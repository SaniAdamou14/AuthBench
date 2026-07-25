from __future__ import annotations

import polars as pl

from authbench.evaluate.campaign import time_to_detection


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": [0, 1, 2, 3, 4],
            "day": [0, 0, 0, 1, 1],
            "time": [100, 200, 300, 86_500, 86_600],
            "is_malicious": [True, True, False, True, False],
            "campaign_id": [1, 1, None, 2, None],
            "score": [0.9, 0.1, 0.5, 0.95, 0.2],
        }
    )


def test_detected_campaign_reports_delay_from_first_event_to_first_alert() -> None:
    frame = _frame()
    # Budget of 1 alert/day: day 0's top alert is event 0 (score 0.9, campaign 1);
    # day 1's top alert is event 3 (score 0.95, campaign 2) — both campaigns detected
    # immediately, at their first event.
    result = time_to_detection(frame, "score", k=1, model_name="test_model")

    assert result.n_total_campaigns == 2
    assert result.n_never_detected == 0
    assert result.detected_delay_seconds == [0.0, 0.0]


def test_never_detected_campaign_counted_separately_not_dropped() -> None:
    frame = _frame()
    # Budget of 0: nothing is ever alerted, so both campaigns go undetected —
    # and must show up in n_never_detected, not silently vanish from the denominator.
    result = time_to_detection(frame, "score", k=0, model_name="test_model")

    assert result.n_total_campaigns == 2
    assert result.n_never_detected == 2
    assert result.detected_delay_seconds == []
    assert result.n_detected == 0
