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
