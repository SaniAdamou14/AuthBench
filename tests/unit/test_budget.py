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


def _fixed_disk_bytes() -> int:
    """Everything that does not scale with the event count: the whole gzip
    source LANL serves in one piece, and train_eval's small fixed outputs."""
    budget = lanl_budget()
    scaling = {"to_parquet", "label_split_features"}
    return sum(s.disk_bytes for s in budget if not any(k in s.stage for k in scaling))


def test_only_the_per_event_terms_scale_with_the_event_count() -> None:
    """Ten times the events costs ten times the *converted* data — but not ten
    times the download, which is one gzip stream LANL serves whole however few
    days a run keeps. Getting that wrong understates a partial run's disk by
    the size of the source file."""
    fixed = _fixed_disk_bytes()
    small = total_disk_bytes(lanl_budget(100_000_000)) - fixed
    large = total_disk_bytes(lanl_budget(1_000_000_000)) - fixed

    assert 9.9 < large / small < 10.1

    download = next(s for s in lanl_budget() if "download" in s.stage).disk_bytes
    assert (
        next(s for s in lanl_budget(10_000_000) if "download" in s.stage).disk_bytes == download
    ), "the download does not shrink with the day window"


def test_the_pipeline_still_costs_several_times_the_download() -> None:
    """docs/scaling.md's headline claim, at the ratio the pipeline now has.

    It used to be about twenty times the download. It is around seven, because
    the per-event terms were recalibrated on LANL itself — the demo-derived
    estimates were 60% high on interim Parquet and 57% high on the feature
    store — and because the store now persists only the columns something
    reads. The download did not change; the pipeline got cheaper. Asserted
    loosely on purpose: the claim worth defending is "planning for the
    download alone is not enough", not any particular multiple."""
    budget = lanl_budget(LANL_TOTAL_EVENTS)
    download = next(s for s in budget if "download" in s.stage)

    assert download.disk_bytes < 10 * 1024**3, "the download itself is under 10 GB"
    assert total_disk_bytes(budget) > 3 * download.disk_bytes


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


def test_disk_is_now_the_binding_constraint_on_a_laptop() -> None:
    """It used to be memory, by a factor of three: `train_eval` loaded whole
    splits and the estimators copied the design matrix. Fitting on a bounded
    sample and scoring by day moved the ceiling far enough that free disk is
    what limits the run again — which is the constraint you can fix by
    plugging in a drive."""
    by_disk = max_events_for_disk(30 * 1024**3)
    by_memory = max_events_for_memory(16 * 1024**3)

    assert by_disk < by_memory


def test_available_disk_and_memory_are_readable_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=100, used=0, free=12345),
    )

    assert available_disk_bytes(tmp_path) == 12345
    assert available_memory_bytes() > 0
