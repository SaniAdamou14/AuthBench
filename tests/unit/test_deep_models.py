"""M4a, M4b and M4c: determinism, row-locality, bounded fitting and the sign of the score."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from authbench.features.sequence import compute_f6  # noqa: E402
from authbench.features.vocabulary import FIRST_TOKEN_ID, OOV_ID, n_tokens  # noqa: E402
from authbench.models.deep import AutoencoderScorer, VAEScorer  # noqa: E402
from authbench.models.sequence import NextEventLSTMScorer  # noqa: E402
from authbench.pipeline.train_eval import build_model_catalog  # noqa: E402

COLUMNS = ["a", "b", "c", "d"]
FAST = {"epochs": 8, "batch_size": 64, "learning_rate": 1e-2}


def _gaussian(n: int, seed: int = 0) -> pl.LazyFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(rng.normal(size=(n, len(COLUMNS))), schema=COLUMNS, orient="row").lazy()


@pytest.mark.parametrize("model_class", [AutoencoderScorer, VAEScorer])
def test_two_fits_with_the_same_seed_give_bit_identical_scores(model_class: type) -> None:
    train, test = _gaussian(600), _gaussian(200, seed=1)
    runs = []
    for _ in range(2):
        model = model_class(COLUMNS, **FAST)
        model.fit(train)
        runs.append(model.score(test).to_numpy())
    np.testing.assert_array_equal(runs[0], runs[1])


def test_vae_scoring_draws_no_random_sample() -> None:
    model = VAEScorer(COLUMNS, **FAST)
    model.fit(_gaussian(600))
    test = _gaussian(200, seed=1)
    np.testing.assert_array_equal(model.score(test).to_numpy(), model.score(test).to_numpy())


@pytest.mark.parametrize("model_class", [AutoencoderScorer, VAEScorer])
def test_scoring_in_pieces_equals_scoring_whole(model_class: type) -> None:
    model = model_class(COLUMNS, score_batch_rows=7, **FAST)
    model.fit(_gaussian(600))
    test = _gaussian(100, seed=2).collect()
    whole = model.score(test.lazy()).to_numpy()
    pieces = np.concatenate([model.score(test.slice(i, 34).lazy()).to_numpy() for i in (0, 34, 68)])
    alone = np.array(
        [model.score(test.slice(i, 1).lazy()).to_numpy()[0] for i in range(0, 100, 11)]
    )
    assert model.scores_row_locally
    np.testing.assert_array_equal(whole, pieces)
    np.testing.assert_array_equal(whole[::11], alone)


@pytest.mark.parametrize("model_class", [AutoencoderScorer, VAEScorer])
def test_row_wise_scoring_evaluates_the_trained_network(model_class: type) -> None:
    """The float64 row-wise forward pass is the torch one, up to torch's float32 rounding."""
    model = model_class(COLUMNS, **FAST)
    model.fit(_gaussian(600))
    test = _gaussian(100, seed=4)
    (inputs,) = model._inputs(test, fitting=False)
    model._module.eval()
    with torch.no_grad():
        expected = model._score(model._module, (inputs,)).to(torch.float64).numpy()
    np.testing.assert_allclose(model.score(test).to_numpy(), expected, rtol=1e-4, atol=1e-5)


def test_the_sequence_model_does_not_claim_row_locality() -> None:
    """M4c still scores in batched float32 torch, whose results move with the batch."""
    assert NextEventLSTMScorer.scores_row_locally is False


def test_fit_is_bounded_by_fit_sample_size() -> None:
    model = AutoencoderScorer(COLUMNS, fit_sample_size=150, **FAST)
    model.fit(_gaussian(600))
    assert model.scaler.n_samples_seen_ == 150
    assert 1 <= len(model.history) <= FAST["epochs"]


@pytest.mark.parametrize("model_class", [AutoencoderScorer, VAEScorer])
def test_a_far_outlier_scores_above_every_inlier(model_class: type) -> None:
    model = model_class(COLUMNS, **FAST)
    model.fit(_gaussian(1000))
    inliers = model.score(_gaussian(200, seed=3)).to_numpy()
    outlier = model.score(
        pl.DataFrame([[25.0] * len(COLUMNS)], schema=COLUMNS, orient="row").lazy()
    )
    assert outlier[0] > inliers.max()


def test_the_catalog_builds_the_autoencoders_from_their_config_files() -> None:
    names = [m.name for m in build_model_catalog(fit_sample_size=None, deep_models=True)]
    assert names[-2:] == ["M4a_autoencoder", "M4b_vae"]
    assert "M4a_autoencoder" not in [m.name for m in build_model_catalog(fit_sample_size=None)]


def _alternating_user(n_events: int) -> pl.LazyFrame:
    return pl.LazyFrame(
        {
            "event_id": list(range(n_events)),
            "time": [10 * i for i in range(n_events)],
            "src_user": ["alice"] * n_events,
            "dst_computer": ["A" if i % 2 == 0 else "B" for i in range(n_events)],
            "auth_type": ["K"] * n_events,
            "success": [True] * n_events,
        },
        schema={
            "event_id": pl.UInt64,
            "time": pl.Int32,
            "src_user": pl.Categorical,
            "dst_computer": pl.Categorical,
            "auth_type": pl.Categorical,
            "success": pl.Boolean,
        },
    )


def test_lstm_finds_an_unseen_destination_more_surprising_than_the_expected_one() -> None:
    vocabulary = ["A", "B"]
    encoded = compute_f6(
        _alternating_user(400),
        dst_vocabulary=vocabulary,
        auth_type_vocabulary=["K"],
        context_length=4,
    ).collect()
    model = NextEventLSTMScorer(n_tokens(vocabulary), epochs=15, batch_size=32, learning_rate=1e-2)
    model.fit(encoded.lazy())

    last = encoded.sort("event_id").tail(1)
    expected = FIRST_TOKEN_ID + (0 if last["dst_code"][0] == FIRST_TOKEN_ID + 1 else 1)
    probes = pl.concat(
        [
            last.with_columns(pl.lit(code, dtype=pl.Int32).alias("dst_code"))
            for code in (expected, OOV_ID)
        ]
    )
    expected_surprise, unseen_surprise = model.score(probes.lazy()).to_numpy()
    assert unseen_surprise > expected_surprise


def test_lstm_rejects_codes_from_a_different_vocabulary() -> None:
    encoded = compute_f6(
        _alternating_user(20),
        dst_vocabulary=["A", "B"],
        auth_type_vocabulary=["K"],
        context_length=2,
    ).collect()
    model = NextEventLSTMScorer(n_tokens=3, epochs=1)
    with pytest.raises(ValueError, match="different vocabulary"):
        model.fit(encoded.lazy())
