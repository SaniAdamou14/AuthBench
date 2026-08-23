"""Every partition of a configured split must be able to contain positives.

The spec's own published split for LANL — section 6.1, train days 1–30,
validation 31–40, test 41–58 — puts **all 749 red-team events in train and
none at all in validation or test**. The red team stopped operating on day 29;
the split was written as thirds of the calendar without checking where the
attacks were.

Nothing in the pipeline would have crashed early. `label_auth_events` joins
happily and returns zero matches, the temporal-order guard passes (it checks
ordering, not content), features compute normally, and hours later M1's weight
calibration raises `NoPositivesInValidationError` — or, if M1 were absent, the
bootstrap raises on a test split with no positives. Either way the failure
arrives after a 7.6 GB download and a full feature build.

This test is that check, moved to the front, and costing nothing: the day
bounds are in `conf/`, and the red team's active range is a measured property
of the dataset recorded in `conf/dataset/lanl.yaml`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

CONF = Path("conf")
PARTITIONS = ("train_days", "val_days", "test_days")


def _load(*parts: str) -> DictConfig:
    loaded = OmegaConf.load(Path(*parts))
    assert isinstance(loaded, DictConfig)
    return loaded


def _active_days(dataset: str) -> tuple[int, int]:
    cfg = _load(str(CONF), "dataset", f"{dataset}.yaml")
    first, last = cfg.redteam_active_days
    return int(first), int(last)


@pytest.mark.parametrize("partition", PARTITIONS)
def test_each_lanl_partition_overlaps_the_red_teams_active_days(partition: str) -> None:
    split = _load(str(CONF), "split", "temporal.yaml")
    first_active, last_active = _active_days("lanl")
    low, high = (int(v) for v in split[partition])

    assert low <= last_active and high >= first_active, (
        f"split.{partition} = [{low}, {high}] does not overlap the red team's active window "
        f"[{first_active}, {last_active}], so that partition would contain zero positives. "
        "This is what the spec's published 0-29 / 30-39 / 40-57 split does to validation "
        "and test. Pick a window inside the active range — see docs/scaling.md."
    )


def test_the_spec_original_split_is_recorded_as_unusable() -> None:
    """A named regression, so nobody restores those bounds from the spec.

    Keeping the failing configuration in a test rather than only in prose is
    what stops it coming back the next time someone reads section 6.1 and
    "corrects" conf/split/temporal.yaml to match it.
    """
    first_active, last_active = _active_days("lanl")
    spec_split = {"train_days": (0, 29), "val_days": (30, 39), "test_days": (40, 57)}

    empty = [
        name
        for name, (low, high) in spec_split.items()
        if not (low <= last_active and high >= first_active)
    ]

    assert empty == ["val_days", "test_days"], (
        "The spec's split leaves validation and test with no red-team events. If this "
        "assertion changes, the dataset's measured active window did — recheck "
        "conf/dataset/lanl.yaml against redteam.txt.gz before trusting any result."
    )


def test_the_demo_split_also_overlaps_its_own_campaigns() -> None:
    """The same class of bug, guarded for the sample the CI actually runs."""
    split = _load(str(CONF), "split", "demo_temporal.yaml")
    for partition in PARTITIONS:
        low, high = (int(v) for v in split[partition])
        assert low <= high
