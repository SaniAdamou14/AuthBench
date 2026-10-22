"""A score must be a function of its row alone, to the last bit.

`train_eval` scores the positives on their own and each test day in chunks, and
counts every day's benign events against the positives' scores. That is exact only
if an event scores the same alone, in any chunk and in the whole split. A tolerance
would hide the failure that matters: BLAS and torch sum in a batch-dependent order,
and a one-ULP difference moves an event across a tie and changes its rank.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from authbench.evaluate.statistics import statistics_from_frame
from authbench.evaluate.summary import EVAL_COLUMNS, score_column
from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.models.rules import RulesScorer
from authbench.pipeline.train_eval import build_model_catalog, fit_models, score_split


def _featurized(n_days: int = 4, per_day: int = 40) -> pl.DataFrame:
    """A frame carrying every column the catalog reads, spread over days."""
    rng = np.random.default_rng(0)
    n = n_days * per_day
    day = np.repeat(np.arange(n_days), per_day)
    time = day * 86_400 + np.tile(np.arange(per_day) * 900, n_days)
    malicious = rng.random(n) < 0.05

    frame = pl.DataFrame(
        {
            "event_id": np.arange(n, dtype=np.int64),
            "time": time.astype(np.int64),
            "day": day.astype(np.int64),
            "is_malicious": malicious,
            "campaign_id": [(i % 3) + 1 if m else None for i, m in enumerate(malicious)],
            "src_user": [f"U{i % 7}" for i in range(n)],
            "src_computer": [f"C{i % 11}" for i in range(n)],
            "dst_computer": [f"C{(i * 3) % 11}" for i in range(n)],
            "auth_type": [["Kerberos", "NTLM", "Negotiate"][i % 3] for i in range(n)],
            "success": rng.random(n) > 0.1,
        }
    )
    # The design matrix, plus the precomputed columns M1's rules read.
    extra = {name: rng.random(n) for name in MODEL_FEATURE_COLUMNS}
    extra["src_user_1h_n_distinct_dst"] = rng.integers(0, 5, n).astype(float)
    extra["dst_computer_1h_n_failures"] = rng.integers(0, 3, n).astype(float)
    extra["user_auth_is_new"] = rng.random(n) < 0.1
    extra["prev_dst_is_src"] = rng.random(n) < 0.2
    extra["hours_since_prev_event_same_user"] = rng.random(n)
    return frame.with_columns([pl.Series(k, v) for k, v in extra.items()])


def _catalog() -> list[object]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return list(build_model_catalog(fit_sample_size=None))
    return list(build_model_catalog(fit_sample_size=None, deep_models=True))


@pytest.mark.parametrize("model", _catalog(), ids=lambda m: str(m.name))
def test_an_event_scores_the_same_alone_in_any_chunk_and_whole(model: object) -> None:
    frame = _featurized()
    model.fit(frame.lazy())  # type: ignore[attr-defined]
    whole = model.score(frame.lazy()).to_numpy()  # type: ignore[attr-defined]
    chunked = np.concatenate(
        [model.score(frame.slice(o, 37).lazy()).to_numpy() for o in range(0, frame.height, 37)]  # type: ignore[attr-defined]
    )
    alone = np.array(
        [model.score(frame.slice(i, 1).lazy()).to_numpy()[0] for i in range(0, frame.height, 9)]  # type: ignore[attr-defined]
    )
    assert model.scores_row_locally  # type: ignore[attr-defined]
    np.testing.assert_array_equal(whole, chunked)
    np.testing.assert_array_equal(whole[::9], alone)


def test_scoring_by_day_gives_the_statistics_of_the_whole_split() -> None:
    frame = _featurized()
    budgets = [1, 5, 20]
    models = build_model_catalog(fit_sample_size=None)
    fit_models(models, frame.lazy(), frame.lazy())
    by_day, skipped = score_split(models, frame.lazy(), budgets=budgets, top_k=20, chunk_rows=13)
    whole = frame.select(EVAL_COLUMNS).with_columns(
        pl.Series(score_column(m.name), m.score(frame.lazy())) for m in models
    )
    expected = statistics_from_frame(
        whole, {m.name: score_column(m.name) for m in models}, budgets=budgets, top_k=20
    )
    assert not skipped
    for name, stats in expected.items():
        actual = by_day[name]
        assert actual.positives.equals(stats.positives), name
        assert actual.top.equals(stats.top), name
        assert np.array_equal(actual.benign_above, stats.benign_above), name
        assert np.array_equal(actual.benign_level, stats.benign_level), name
        assert actual.tie_exposure == stats.tie_exposure, name


def test_a_model_that_is_not_row_local_is_refused() -> None:
    class Batchy(RulesScorer):
        scores_row_locally = False

    with pytest.raises(ValueError, match="not row-local"):
        score_split([Batchy()], _featurized().lazy(), budgets=[1], top_k=1)


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


def test_r2_trains_without_the_red_team_and_r1_with_it() -> None:
    from authbench.pipeline.train_eval import R1, R2, regime_train

    frame = _featurized()
    assert regime_train(frame.lazy(), R1).collect().equals(frame)
    clean = regime_train(frame.lazy(), R2).collect()
    assert not clean["is_malicious"].any()
    assert clean.height == frame.height - int(frame["is_malicious"].sum())
    with pytest.raises(ValueError, match="Unknown regime"):
        regime_train(frame.lazy(), "r3")
