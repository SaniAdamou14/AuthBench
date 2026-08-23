"""The sweep-based distinct count must equal the self-join it replaces.

`causal_distinct_count` exists because `causal_prior_events` cannot run at LANL
scale — 34.6 billion pairs for one day. A replacement that is merely *fast* is
worth nothing: it has to produce the same number the exact self-join would, on
the cases that are easy to get wrong. Ties on the window edge, a partner seen
once long ago, a partner seen continuously, an entity's first-ever event.

The self-join is the oracle here, run on frames small enough for it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.features.causal import causal_distinct_count, causal_prior_events

WINDOW = 3600


def _reference(frame: pl.DataFrame, window_seconds: int = WINDOW) -> dict[int, int]:
    """Distinct partners in the strictly-prior window, via the exact self-join."""
    joined = causal_prior_events(
        frame.lazy(),
        entity_col="src_user",
        time_col="time",
        id_col="event_id",
        carry_cols=["dst_computer"],
        window_seconds=window_seconds,
        namespace="ref",
    )
    counts = (
        joined.group_by("_current_id").agg(pl.col("dst_computer").n_unique().alias("n")).collect()
    )
    out = dict(zip(counts["_current_id"].to_list(), counts["n"].to_list(), strict=True))
    # Events with no prior event in the window are absent from the join, and
    # their distinct count is zero.
    return {int(i): int(out.get(i, 0)) for i in frame["event_id"].to_list()}


def _sweep(frame: pl.DataFrame, window_seconds: int = WINDOW) -> dict[int, int]:
    result = causal_distinct_count(
        frame.lazy(),
        entity_col="src_user",
        partner_col="dst_computer",
        time_col="time",
        id_col="event_id",
        window_seconds=window_seconds,
        out_col="n",
    ).collect()
    return dict(
        zip(
            [int(i) for i in result["event_id"].to_list()],
            [int(v) for v in result["n"].to_list()],
            strict=True,
        )
    )


def _frame(rows: list[tuple[str, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": list(range(len(rows))),
            "src_user": [r[0] for r in rows],
            "dst_computer": [r[1] for r in rows],
            "time": [r[2] for r in rows],
        }
    )


def test_a_partner_that_stays_active_is_counted_once_not_repeatedly() -> None:
    """The case a naive "count new pairs in the window" rolling sum gets wrong:
    a partner seen continuously never re-enters, so a sum over introductions
    would drop it while it is plainly still in the window."""
    frame = _frame([("u", "a", t) for t in (0, 600, 1200, 1800)] + [("u", "b", 2000)])

    sweep = _sweep(frame)

    # At t=1800: 'a' seen at 0/600/1200, all within the hour -> exactly 1 distinct.
    assert sweep[3] == 1
    # At t=2000: 'a' and 'b'... 'b' is the current event, so only 'a' is prior.
    assert sweep[4] == 1
    assert sweep == _reference(frame)


def test_a_partner_falling_out_of_the_window_stops_being_counted() -> None:
    frame = _frame([("u", "a", 0), ("u", "b", 100), ("u", "b", 3500), ("u", "b", 5000)])

    sweep = _sweep(frame)

    assert sweep[1] == 1  # only 'a' is prior
    assert sweep[2] == 2  # 'a' (t=0, still within 3600) and 'b'
    assert sweep[3] == 1  # 'a' has aged out; only 'b'
    assert sweep == _reference(frame)


def test_the_window_edge_is_half_open_exactly_like_the_rolling_primitives() -> None:
    """`[t - W, t)`: closed on the left, open on the right.

    So an event exactly `W` seconds old is **in** the window, and one a second
    older is not. Both ends are asserted, because getting either wrong shifts
    every count by one in a way no aggregate would reveal.
    """
    frame = _frame([("u", "a", 0), ("u", "b", WINDOW), ("u", "c", WINDOW + 1)])

    sweep = _sweep(frame)

    assert sweep[0] == 0, "the first event of an entity has nothing prior"
    assert sweep[1] == 1, "an event at exactly t-W is inside the left-closed window"
    assert sweep[2] == 1, "'a' at t=0 is now W+1 seconds old and has aged out; only 'b' remains"
    assert sweep == _reference(frame)


def test_entities_never_see_each_other() -> None:
    frame = _frame([("u", "a", 0), ("v", "b", 10), ("v", "c", 20), ("u", "d", 30)])

    sweep = _sweep(frame)

    assert sweep[3] == 1  # u saw only 'a'
    assert sweep[2] == 1  # v saw only 'b'
    assert sweep == _reference(frame)


@pytest.mark.parametrize("seed", range(8))
def test_the_sweep_matches_the_self_join_on_random_frames(seed: int) -> None:
    """Property check across many shapes: heavy ties, repeated partners,
    several entities, timestamps clustered inside and across the window."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 160))
    rows = [
        (
            f"u{rng.integers(0, 3)}",
            f"c{rng.integers(0, 5)}",
            int(rng.integers(0, 4 * WINDOW)),
        )
        for _ in range(n)
    ]
    frame = _frame(rows)

    assert _sweep(frame) == _reference(frame)


def test_it_scales_where_the_self_join_cannot() -> None:
    """One busy entity: 20,000 events in an hour. The self-join would build
    400 million pairs for this alone; the sweep is linearithmic."""
    n = 20_000
    rng = np.random.default_rng(0)
    frame = _frame(
        [("busy$", f"c{rng.integers(0, 50)}", int(t)) for t in np.sort(rng.integers(0, WINDOW, n))]
    )

    sweep = _sweep(frame)

    assert len(sweep) == n
    # Every partner drawn from 50, all inside one window: the count climbs
    # toward 50 and never exceeds it.
    assert max(sweep.values()) <= 50
    assert sweep[int(frame["event_id"][-1])] >= 45
