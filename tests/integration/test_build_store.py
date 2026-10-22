"""`build_store` writes, day by day, the same features as featurising each partition whole."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf
from polars.testing import assert_frame_equal

from authbench.features import FEATURE_STORE_COLUMNS
from authbench.features.event import fit_frequency_encoding
from authbench.features.temporal import calibrate_night_window
from authbench.label.redteam_join import attach_campaign_id, group_into_campaigns, label_auth_events
from authbench.parse.clean import clean_auth, clean_redteam
from authbench.pipeline.build_features import (
    StoreInputs,
    build_store,
    featurize,
    split_ranges,
    universe_filter,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def demo() -> tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame]:
    dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "demo.yaml")

    def scan(path: str, columns: list[str]) -> pl.LazyFrame:
        return pl.scan_csv(ROOT / path, has_header=False, new_columns=columns).with_columns(
            pl.col("time").cast(pl.Int64)
        )

    auth, _ = clean_auth(scan(str(dataset.auth_path), list(dataset.auth_columns)))
    redteam, _ = clean_redteam(scan(str(dataset.redteam_path), list(dataset.redteam_columns)))
    campaigns = group_into_campaigns(redteam, gap_hours=float(dataset.campaign_gap_hours))
    return auth.collect(), redteam, campaigns


def _inputs(demo: tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame], train_end: int) -> StoreInputs:
    auth, redteam, campaigns = demo
    train = auth.filter(pl.col("time") < train_end).lazy()
    return StoreInputs(
        read_day=lambda d: auth.filter(pl.col("day") == d) if d <= auth["day"].max() else None,  # type: ignore[operator]
        redteam=redteam,
        campaigns=campaigns,
        night_window=calibrate_night_window(train),
        frequency_encoding=fit_frequency_encoding(train),
    )


def _reference(
    demo: tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame],
    inputs: StoreInputs,
    first_day: int,
    start: int,
    end: int,
) -> pl.DataFrame:
    auth, redteam, campaigns = demo
    window = attach_campaign_id(
        label_auth_events(auth.filter(pl.col("day") >= first_day).lazy(), redteam), campaigns
    )
    features_cfg = OmegaConf.load(ROOT / "conf" / "features" / "base.yaml")
    whole = featurize(window, features_cfg, inputs.night_window, inputs.frequency_encoding)
    return (
        whole.filter(pl.col("time").is_between(start, end, closed="left"))
        .select(FEATURE_STORE_COLUMNS)
        .collect()
    )


def _written(out: Path, split: str) -> pl.DataFrame:
    return pl.read_parquet(out / split / "*.parquet")


def test_window_history_matches_each_partition_featurised_whole(
    demo: tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame], tmp_path: Path
) -> None:
    ranges = split_ranges(OmegaConf.load(ROOT / "conf" / "split" / "demo_temporal.yaml"))
    inputs = _inputs(demo, ranges["train"][1])
    (tmp_path / "test").mkdir()
    pl.DataFrame({"stale": [1]}).write_parquet(tmp_path / "test" / "day=999.parquet")

    summaries = build_store(inputs, ranges, tmp_path, history="window", warmup_days=2)

    assert set(summaries) == {"train", "val", "test"}
    assert sorted(p.name for p in (tmp_path / "test").iterdir()) == [
        "day=011.parquet",
        "day=012.parquet",
        "day=013.parquet",
    ]
    for split, (start, end) in ranges.items():
        first_day = max(0, start // 86_400 - 2)
        expected = _reference(demo, inputs, first_day, start, end).sort("event_id")
        assert_frame_equal(_written(tmp_path, split).sort("event_id"), expected, check_exact=True)


def test_cumulative_history_with_hour_bounds_reads_every_earlier_event(
    demo: tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame], tmp_path: Path
) -> None:
    ranges = split_ranges(OmegaConf.create({"train_hours": [0, 41], "test_hours": [41, 336]}))
    inputs = _inputs(demo, ranges["train"][1])

    build_store(inputs, ranges, tmp_path, history="cumulative")

    # Day 1 straddles hour 41, so its events are split between the two partitions.
    assert (tmp_path / "train" / "day=001.parquet").exists()
    assert (tmp_path / "test" / "day=001.parquet").exists()
    assert not (tmp_path / "val").exists()
    for split, (start, end) in ranges.items():
        written = _written(tmp_path, split)
        assert written["time"].min() >= start and written["time"].max() < end  # type: ignore[operator]
        expected = _reference(demo, inputs, 0, start, end).sort("event_id")
        assert_frame_equal(written.sort("event_id"), expected, check_exact=True)


def test_split_ranges_reject_overlapping_or_unordered_partitions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_ranges(OmegaConf.create({"train_hours": [0, 50], "test_hours": [41, 100]}))
    with pytest.raises(ValueError, match="train and test"):
        split_ranges(OmegaConf.create({"test_days": [3, 4]}))


def test_universes_are_nested(demo: tuple[pl.DataFrame, pl.LazyFrame, pl.LazyFrame]) -> None:
    auth = demo[0]
    sizes = [
        auth.filter(universe_filter(u)).height for u in ("all", "remote_logon", "remote_logon_user")
    ]
    assert sizes[0] > sizes[1] > sizes[2] > 0
    remote = auth.filter(universe_filter("remote_logon"))
    assert (remote["src_computer"].cast(pl.Utf8) != remote["dst_computer"].cast(pl.Utf8)).all()
    with pytest.raises(ValueError, match="Unknown universe"):
        universe_filter("local")
