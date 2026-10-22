"""The staffing model is arithmetic on published inputs; these pin the arithmetic."""

from __future__ import annotations

import pytest

from authbench.evaluate.staffing import (
    SOC_ALERTS_HANDLED_PER_DAY,
    TRIAGE_MINUTES_PER_ANALYST_DAY,
    staffing,
    staffing_table,
)


def test_ten_alerts_a_day_is_about_one_analyst_at_ten_to_twenty_minutes_each() -> None:
    row = staffing(10)
    assert row.analysts_central == pytest.approx((100 / 162, 200 / 162))
    assert row.analysts_fastest < row.analysts_central[0] < row.analysts_central[1]
    assert row.analysts_central[1] < row.analysts_slowest


def test_the_inputs_are_the_published_ones() -> None:
    assert pytest.approx(162) == TRIAGE_MINUTES_PER_ANALYST_DAY
    assert pytest.approx(4484 * 0.33) == SOC_ALERTS_HANDLED_PER_DAY


def test_analysts_scale_linearly_with_the_budget_and_the_table_rounds_them() -> None:
    assert staffing(100).analysts_slowest == pytest.approx(10 * staffing(10).analysts_slowest)
    assert staffing_table([10])[0]["analysts_at_30_min"] == round(staffing(10).analysts_slowest, 2)
