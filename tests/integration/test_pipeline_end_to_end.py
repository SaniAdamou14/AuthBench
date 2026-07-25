"""Integration test: ingest -> clean -> label -> split -> features -> model
-> evaluate, wired together exactly like `authbench.cli.demo`, but against a
small in-memory-generated sample so the suite stays fast. The CLI's own
`make demo` path (full default-sized sample) is exercised separately in CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from authbench.evaluate.budget import compute_budget_curve
from authbench.evaluate.metrics import auc_pr
from authbench.features.event import compute_f1
from authbench.features.history import compute_f2
from authbench.features.novelty import compute_f3
from authbench.features.temporal import calibrate_night_window, compute_f4
from authbench.label.redteam_join import attach_campaign_id, group_into_campaigns, label_auth_events
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.stats import PairRarityScorer
from authbench.parse.clean import clean_auth, clean_redteam
from authbench.parse.schema import RAW_AUTH_COLUMNS, RAW_REDTEAM_COLUMNS
from authbench.split.temporal import (
    TemporalSplitConfig,
    get_test_split,
    get_train_split,
    get_val_split,
    verify_temporal_order,
)


def _generate_small_demo(tmp_path: Path) -> tuple[Path, Path]:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from generate_demo_data import generate

    auth_lines, redteam_lines = generate(
        n_days=6,
        n_users=20,
        n_machine_accounts=5,
        n_computers=15,
        events_per_user_per_day=3.0,
        seed=7,
    )
    auth_path = tmp_path / "auth.txt"
    redteam_path = tmp_path / "redteam.txt"
    auth_path.write_text("\n".join(auth_lines) + "\n")
    redteam_path.write_text("\n".join(redteam_lines) + "\n")
    return auth_path, redteam_path


def test_full_pipeline_runs_end_to_end_on_small_sample(tmp_path: Path) -> None:
    auth_path, redteam_path = _generate_small_demo(tmp_path)

    raw_auth = pl.scan_csv(auth_path, has_header=False, new_columns=RAW_AUTH_COLUMNS)
    typed_auth, _ = clean_auth(raw_auth)

    raw_redteam = pl.scan_csv(redteam_path, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    typed_redteam, _ = clean_redteam(raw_redteam)

    campaigns = group_into_campaigns(typed_redteam, gap_hours=24.0)
    labeled = label_auth_events(typed_auth, typed_redteam)
    labeled = attach_campaign_id(labeled, campaigns)

    n_positives = labeled.filter(pl.col("is_malicious")).select(pl.len()).collect().item()
    assert n_positives > 0, "the generator's injected campaigns must produce at least one positive"

    split_config = TemporalSplitConfig(train_days=(0, 2), val_days=(3, 3), test_days=(4, 5))
    train = get_train_split(labeled, split_config)
    val = get_val_split(labeled, split_config)
    test = get_test_split(labeled, split_config)
    verify_temporal_order(train, val, test)

    night_window = calibrate_night_window(train)

    def featurize(frame: pl.LazyFrame) -> pl.LazyFrame:
        frame = compute_f1(frame)
        frame = compute_f2(frame, entities=["src_user"], windows_hours=[1, 24])
        frame = compute_f3(frame)
        frame = compute_f4(frame, night_window)
        return frame

    train_feat = featurize(train).collect()
    test_feat = featurize(test).collect()

    assert train_feat.height > 0
    assert test_feat.height > 0

    y_test = test_feat["is_malicious"].to_numpy()

    for model in [RandomScorer(), AlwaysFailScorer(), PairRarityScorer()]:
        model.fit(train_feat.lazy())
        scores = model.score(test_feat.lazy())
        assert len(scores) == test_feat.height

        ap = auc_pr(y_test, scores.to_numpy())
        assert 0.0 <= ap <= 1.0

        scored_frame = test_feat.with_columns(pl.Series("_score", scores))
        curve = compute_budget_curve(scored_frame, "_score", model.name)
        for recall in [*curve.event_recall, *curve.campaign_recall]:
            assert 0.0 <= recall <= 1.0


def test_random_baseline_scores_are_reproducible_with_same_seed() -> None:
    frame = pl.LazyFrame({"event_id": list(range(100))})
    scores_a = RandomScorer(seed=123).score(frame)
    scores_b = RandomScorer(seed=123).score(frame)
    assert np.allclose(scores_a.to_numpy(), scores_b.to_numpy())
