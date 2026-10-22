"""Scoring a split in chunks must not change a single score.

This is the change that lets `train_eval` run on a real dataset, and it is
exactly the kind of optimisation that silently alters results: a model whose
score depends on other rows produces perfectly plausible numbers when its
history is reset at every chunk boundary. So the equivalence is asserted, per
model, against the whole-split answer — and the one model that is *not*
equivalent is asserted to be excluded from the chunked path.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from authbench.evaluate.summary import score_column
from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.models.rules import RulesScorer
from authbench.pipeline.train_eval import build_model_catalog, fit_and_score_all


def _featurized(n_days: int = 4, per_day: int = 40) -> pl.DataFrame:
    """A frame carrying every column the catalog reads, spread over days."""
    rng = np.random.default_rng(0)
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    time = day * 86_400 + np.tile(np.arange(per_day) * 900, n_days)

    frame = pl.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "time": time.astype(np.int64),
            "day": day.astype(np.int64),
            "is_malicious": rng.random(n) < 0.05,
            "campaign_id": [
                (i % 3) + 1 if malicious else None
                for i, malicious in enumerate(rng.random(n) < 0.05)
            ],
            "src_user": [f"U{i % 7}" for i in range(n)],
            "src_computer": [f"C{i % 11}" for i in range(n)],
            "dst_computer": [f"C{(i * 3) % 11}" for i in range(n)],
            "auth_type": [["Kerberos", "NTLM", "Negotiate"][i % 3] for i in range(n)],
            "success": rng.random(n) > 0.1,
        }
    )
    # The design matrix, plus the two precomputed columns M1's rules read.
    extra = {name: rng.random(n) for name in MODEL_FEATURE_COLUMNS}
    extra["src_user_1h_n_distinct_dst"] = rng.integers(0, 5, n).astype(float)
    extra["dst_computer_1h_n_failures"] = rng.integers(0, 3, n).astype(float)
    extra["user_auth_is_new"] = rng.random(n) < 0.1
    extra["prev_dst_is_src"] = rng.random(n) < 0.2
    extra["hours_since_prev_event_same_user"] = rng.random(n)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def test_chunked_scoring_reproduces_whole_split_scoring_exactly() -> None:
    frame = _featurized()
    train, val, test = frame, frame, frame

    whole, whole_skipped = fit_and_score_all(
        build_model_catalog(fit_sample_size=None),
        train.lazy(),
        val.lazy(),
        test.lazy(),
        chunk_rows=None,
    )
    chunked, chunked_skipped = fit_and_score_all(
        build_model_catalog(fit_sample_size=None),
        train.lazy(),
        val.lazy(),
        test.lazy(),
        chunk_rows=37,  # deliberately not aligned to any day boundary
    )

    assert not whole_skipped and not chunked_skipped, "no model should fail on this frame"
    assert whole.height == chunked.height == frame.height
    assert whole["event_id"].to_list() == chunked["event_id"].to_list(), (
        "the chunked path must preserve row order — every caller attaches scores by position"
    )

    for name in [m.name for m in build_model_catalog()]:  # type: ignore[attr-defined]
        column = score_column(name)
        np.testing.assert_allclose(
            whole[column].to_numpy(),
            chunked[column].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"{name} scores differ between the whole-split and chunked paths",
        )


def test_every_model_in_the_catalog_is_row_local() -> None:
    """M1 joined the chunked path when R6 and R7 became stored features and its scale a training fit."""
    catalog = build_model_catalog()
    transductive = {"M0a_random", "M3b_ecod"}
    assert all(m.scores_row_locally for m in catalog if m.name not in transductive)
    rules = next(m for m in catalog if isinstance(m, RulesScorer))
    assert rules.scores_row_locally is True


def test_m1_scored_by_day_equals_m1_scored_whole() -> None:
    frame = _featurized()
    model = RulesScorer()
    model.fit(frame.lazy())
    whole = model.score(frame.lazy()).to_numpy()
    per_day = np.concatenate(
        [
            model.score(frame.filter(pl.col("day") == day).lazy()).to_numpy()
            for day in sorted(frame["day"].unique().to_list())
        ]
    )
    np.testing.assert_array_equal(whole, per_day)
