"""The demo sample must carry positives in *every* partition.

Regression guard for a bug that was invisible from the outside: the generator
placed its campaigns on a train day and a test day only, leaving the
validation window empty. M1's weight calibration maximizes AUC-PR on
validation, `average_precision_score` returns 0.0 for every candidate when
there are no positives, so the random search kept its first arbitrary draw —
and the benchmark's central heuristic competitor was reported at a score it
had never actually been tuned for. Nothing crashed; the number was just wrong.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from authbench.evaluate.stats_tests import MIN_CAMPAIGNS_FOR_SIGNIFICANCE
from authbench.models.rules import RULES, NoPositivesInValidationError, RulesScorer
from authbench.split.temporal import TemporalSplitConfig

DEMO_REDTEAM = "data/demo/redteam_demo.txt"
DEMO_SPLIT_CONF = "conf/split/demo_temporal.yaml"
SECONDS_PER_DAY = 86_400


def _redteam_days() -> set[int]:
    frame = pl.read_csv(
        DEMO_REDTEAM,
        has_header=False,
        new_columns=["time", "user_at_domain", "src_computer", "dst_computer"],
    )
    return set((frame["time"] // SECONDS_PER_DAY).to_list())


def _redteam_frame() -> pl.DataFrame:
    return pl.read_csv(
        DEMO_REDTEAM,
        has_header=False,
        new_columns=["time", "user_at_domain", "src_computer", "dst_computer"],
    ).with_columns((pl.col("time") // SECONDS_PER_DAY).alias("day"))


def _campaigns_in(partition: str) -> int:
    """Campaigns are grouped per `user@domain` (`label.redteam_join`), and the
    generator gives each campaign its own user, so distinct users in a
    partition is its campaign count."""
    config = TemporalSplitConfig.from_hydra(OmegaConf.load(DEMO_SPLIT_CONF))
    start, end = getattr(config, f"{partition}_days")
    window = _redteam_frame().filter(pl.col("day").is_between(start, end))
    return int(window["user_at_domain"].n_unique())


@pytest.mark.parametrize("partition", ["train", "val", "test"])
def test_every_demo_partition_contains_redteam_activity(partition: str) -> None:
    config = TemporalSplitConfig.from_hydra(OmegaConf.load(DEMO_SPLIT_CONF))
    start, end = getattr(config, f"{partition}_days")
    days_in_partition = {d for d in _redteam_days() if start <= d <= end}

    assert days_in_partition, (
        f"The demo sample has no red-team events in the {partition} partition "
        f"(days {start}-{end}). Regenerate it with scripts/generate_demo_data.py — "
        "an empty validation window silently reduces M1's weight calibration to a "
        "single arbitrary random draw."
    )


def test_the_test_split_carries_enough_campaigns_to_estimate_significance() -> None:
    """The second way this sample can cripple the benchmark without crashing.

    Campaigns, not events, are the independent unit of the campaign-stratified
    bootstrap: they *are* its sample size. With one campaign in the test split
    no resample can vary the campaign composition, so no pairwise difference
    can change sign, every p-value lands on the resolution floor `2/(R+1)`, and
    the entire comparison table reads "outperforms" off a single attack. The
    sample used to have exactly one, and the demo reported 21/21 significant.

    `MIN_CAMPAIGNS_FOR_SIGNIFICANCE` now withholds the verdict below two, which
    turns the silent wrong answer into a visible refusal — but a sample that
    triggers it can never demonstrate what the demo exists to demonstrate.
    """
    n_campaigns = _campaigns_in("test")

    assert n_campaigns >= MIN_CAMPAIGNS_FOR_SIGNIFICANCE, (
        f"The demo test split holds {n_campaigns} campaign(s); the bootstrap needs at "
        f"least {MIN_CAMPAIGNS_FOR_SIGNIFICANCE} for significance to be estimable at all. "
        "Raise _CAMPAIGNS_PER_PARTITION['test'] in scripts/generate_demo_data.py."
    )


def test_validation_carries_more_than_one_campaign_for_m1_calibration() -> None:
    """M1's weight search maximizes AUC-PR on validation. One campaign there
    makes that objective nearly degenerate: almost every weight vector scores
    the same handful of events identically, so the search returns something
    arbitrary that still looks calibrated."""
    assert _campaigns_in("val") >= 2


def test_calibrate_weights_rejects_a_validation_set_with_no_positives() -> None:
    """The guard that makes the failure above loud rather than plausible."""
    n = 64
    rng = np.random.default_rng(0)
    features = pl.LazyFrame(
        {
            "event_id": np.arange(n),
            "time": np.arange(n) * 60,
            "src_user": ["U1"] * n,
            "auth_type": ["Kerberos"] * n,
            "src_computer": ["C1"] * n,
            "dst_computer": ["C2"] * n,
            "pair_is_new": rng.integers(0, 2, n).astype(bool),
            "src_user_1h_n_distinct_dst": rng.integers(0, 5, n),
            "dst_computer_1h_n_failures": rng.integers(0, 3, n),
            "is_success": rng.integers(0, 2, n).astype(bool),
            "hour_deviation_from_profile": rng.random(n),
            "src_user_is_machine": np.zeros(n, dtype=bool),
            "pair_global_rarity": rng.random(n),
        }
    )
    all_negative = pl.Series("is_malicious", np.zeros(n, dtype=bool))

    scorer = RulesScorer()
    scorer.fit(features)
    with pytest.raises(NoPositivesInValidationError, match="validation"):
        scorer.calibrate_weights(features, all_negative, n_trials=5)

    # The weights must be left at their uniform prior, not half-mutated.
    assert scorer.weights == pytest.approx({r.id: 1.0 / len(RULES) for r in RULES})
