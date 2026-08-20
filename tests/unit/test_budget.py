"""The cost model behind `authbench preflight`.

Its constants are measured, not guessed (see `docs/scaling.md`), but a cost
model nobody checks is a cost model that quietly stops matching the pipeline
it describes. These tests pin the invariants: it scales with event count, it
inverts consistently, and it errs high rather than low.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from authbench.ingest.budget import (
    LANL_TOTAL_EVENTS,
    available_disk_bytes,
    available_memory_bytes,
    lanl_budget,
    max_events_for_disk,
    max_events_for_memory,
    peak_rss_bytes,
    total_disk_bytes,
)


def test_the_budget_covers_every_stage_that_writes_data() -> None:
    stages = [stage.stage for stage in lanl_budget()]

    assert any("download" in s for s in stages)
    assert any("to_parquet" in s for s in stages)
    assert any("label_split_features" in s for s in stages)
    assert any("train_eval" in s for s in stages)


def test_disk_is_cumulative_and_memory_is_the_largest_single_stage() -> None:
    budget = lanl_budget()

    assert total_disk_bytes(budget) == sum(s.disk_bytes for s in budget)
    assert peak_rss_bytes(budget) == max(s.peak_rss_bytes for s in budget)


def test_cost_scales_with_event_count() -> None:
    """At benchmark scale the per-event terms dominate the fixed overhead, so
    ten times the events costs very nearly ten times the disk."""
    small = total_disk_bytes(lanl_budget(100_000_000))
    large = total_disk_bytes(lanl_budget(1_000_000_000))

    assert 9.9 < large / small < 10.1


def test_full_lanl_needs_far_more_than_the_download(  # the headline claim of docs/scaling.md
) -> None:
    budget = lanl_budget(LANL_TOTAL_EVENTS)
    download = next(s for s in budget if "download" in s.stage)

    assert download.disk_bytes < 10 * 1024**3, "the download itself is under 10 GB"
    assert total_disk_bytes(budget) > 100 * 1024**3, "the pipeline around it is not"


def test_max_events_for_disk_inverts_the_disk_budget() -> None:
    """Whatever `max_events_for_disk` reports must actually fit."""
    available = 40 * 1024**3
    fits = max_events_for_disk(available)

    # `max_events_for_disk` excludes train_eval's fixed overhead and keeps a
    # 5 GB margin, so the answer is comfortably inside the full budget.
    assert total_disk_bytes(lanl_budget(fits)) < available
    assert total_disk_bytes(lanl_budget(int(fits * 1.5))) > available


def test_max_events_for_disk_never_goes_negative() -> None:
    assert max_events_for_disk(0) == 0
    assert max_events_for_disk(1024) == 0


def test_memory_is_the_binding_constraint_on_a_laptop() -> None:
    """16 GB of RAM caps the run harder than 21 GB of free disk does — the
    reason `docs/scaling.md` says the fix is an out-of-core `train_eval`, not
    a bigger drive."""
    by_disk = max_events_for_disk(21 * 1024**3)
    by_memory = max_events_for_memory(16 * 1024**3)

    assert by_memory < by_disk


def test_available_disk_and_memory_are_readable_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=100, used=0, free=12345),
    )

    assert available_disk_bytes(tmp_path) == 12345
    assert available_memory_bytes() > 0
