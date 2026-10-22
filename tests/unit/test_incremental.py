"""Day-by-day featurisation must equal whole-window featurisation, value for value."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from omegaconf import OmegaConf

from authbench.features import FEATURE_STORE_COLUMNS
from authbench.features.event import fit_frequency_encoding
from authbench.features.incremental import featurize_window
from authbench.features.temporal import calibrate_night_window
from authbench.label.redteam_join import attach_campaign_id, group_into_campaigns, label_auth_events
from authbench.parse.clean import clean_auth, clean_redteam
from authbench.pipeline.build_features import featurize

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def labeled() -> pl.DataFrame:
    dataset = OmegaConf.load(ROOT / "conf" / "dataset" / "demo.yaml")
    auth = pl.scan_csv(
        ROOT / str(dataset.auth_path), has_header=False, new_columns=list(dataset.auth_columns)
    ).with_columns(pl.col("time").cast(pl.Int64))
    redteam = pl.scan_csv(
        ROOT / str(dataset.redteam_path),
        has_header=False,
        new_columns=list(dataset.redteam_columns),
    ).with_columns(pl.col("time").cast(pl.Int64))
    typed_auth, _ = clean_auth(auth)
    typed_redteam, _ = clean_redteam(redteam)
    campaigns = group_into_campaigns(typed_redteam, gap_hours=float(dataset.campaign_gap_hours))
    return attach_campaign_id(label_auth_events(typed_auth, typed_redteam), campaigns).collect()


@pytest.mark.parametrize("first_day,last_day", [(0, 13), (3, 9)])
def test_day_by_day_equals_whole_window(
    labeled: pl.DataFrame, first_day: int, last_day: int
) -> None:
    window = labeled.filter(pl.col("day").is_between(first_day, last_day))
    train = labeled.filter(pl.col("day").is_between(0, 7)).lazy()
    night_window = calibrate_night_window(train)
    encoding = fit_frequency_encoding(train)
    features_cfg = OmegaConf.load(ROOT / "conf" / "features" / "base.yaml")

    whole = featurize(window.lazy(), features_cfg, night_window, encoding).collect()
    by_day = featurize_window(
        [window.filter(pl.col("day") == d) for d in range(first_day, last_day + 1)],
        night_window,
        encoding,
    )

    columns = [c for c in whole.columns if c in by_day.columns]
    assert set(FEATURE_STORE_COLUMNS) <= set(columns)
    expected = whole.select(columns).sort("event_id")
    actual = by_day.select(columns).sort("event_id")
    assert actual.height == expected.height
    for column in columns:
        assert actual[column].to_list() == expected[column].to_list(), column
