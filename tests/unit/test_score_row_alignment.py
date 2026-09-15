"""A model's score must belong to the row it was computed from.

Both callers attach scores positionally — `frame.with_columns(pl.Series(col,
model.score(frame)))` in `cli.demo` and in `train_eval.fit_and_score_all`. That
is only correct if `score()` returns one value per input row, *in the input's
own order*. Nothing enforced it. Several models compute their features on
re-sorted frames and join back (`rules.RulesScorer` does it twice, for R6 and
R7), and a join that quietly reordered its output would attribute every event's
score to a different event — with a results table that still looked entirely
plausible.

`models.rules` guards its own two joins with `maintain_order="left"` and says
why. This is the same guarantee, asserted from outside, for every model in the
catalog at once: shuffle the frame, score it again, and every event must carry
the score it had before.
"""

from __future__ import annotations

import polars as pl
import pytest
from tests.unit.test_chunked_scoring import _featurized

from authbench.cli import demo_model_catalog
from authbench.models.base import BaseAnomalyScorer
from authbench.models.floors import RandomScorer
from authbench.pipeline.train_eval import build_model_catalog


def _scores_by_event(model: object, frame: pl.DataFrame) -> dict[int, float]:
    scores = model.score(frame.lazy())  # type: ignore[attr-defined]
    assert len(scores) == frame.height, (
        f"{model.name} returned {len(scores)} scores for {frame.height} rows"
    )  # type: ignore[attr-defined]
    return dict(zip(frame["event_id"].to_list(), scores.to_list(), strict=True))


def _catalog() -> list[object]:
    """Every model either entry point runs, de-duplicated by name."""
    seen: dict[str, object] = {}
    for model in [*build_model_catalog(fit_sample_size=None), *demo_model_catalog()]:
        seen.setdefault(model.name, model)  # type: ignore[attr-defined]
    return list(seen.values())


@pytest.mark.parametrize(
    ("catalog_name", "catalog"),
    [
        ("train_eval.build_model_catalog", build_model_catalog(fit_sample_size=None)),
        ("cli.demo_model_catalog", demo_model_catalog()),
    ],
    ids=["train_eval", "demo"],
)
def test_model_names_are_unique_within_a_catalog(catalog_name: str, catalog: list[object]) -> None:
    """Two models sharing a name would silently become one.

    Scores land on a shared frame under `score_column(name)`, and both entry
    points attach them with `with_columns` — which *replaces* an existing
    column rather than refusing. A duplicate name would therefore drop one
    model's scores, evaluate the survivor twice under two labels, and report a
    comparison of a model against itself as a genuine pair. Nothing about the
    output would look wrong.
    """
    names = [model.name for model in catalog]  # type: ignore[attr-defined]
    duplicates = {name for name in names if names.count(name) > 1}

    assert not duplicates, f"{catalog_name} has duplicate model names: {duplicates}"


@pytest.mark.parametrize("model", _catalog(), ids=lambda m: str(m.name))
def test_every_catalog_model_satisfies_the_scorer_contract(model: object) -> None:
    """`AnomalyScorer` is the one contract a model must meet to enter the
    results table (spec 3.5). Checked structurally rather than by inheritance,
    because the protocol is deliberately structural.
    """
    assert isinstance(getattr(model, "name", None), str) and model.name, (  # type: ignore[attr-defined]
        "a model needs a name: it is the key its scores are stored under"
    )
    assert isinstance(getattr(model, "requires_labels", None), bool)
    assert isinstance(getattr(model, "scores_row_locally", None), bool)
    assert callable(getattr(model, "fit", None))
    assert callable(getattr(model, "score", None))


@pytest.mark.parametrize("model", _catalog(), ids=lambda m: str(m.name))
def test_every_score_follows_its_own_row_through_a_shuffle(model: object) -> None:
    frame = _featurized()
    model.fit(frame.lazy())  # type: ignore[attr-defined]

    if isinstance(model, RandomScorer):
        pytest.skip("M0a's score is positional by construction — see the test below.")

    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=19)
    assert shuffled["event_id"].to_list() != frame["event_id"].to_list(), "the shuffle did nothing"

    assert _scores_by_event(model, frame) == pytest.approx(_scores_by_event(model, shuffled))


def test_the_random_floor_is_the_one_model_whose_score_is_not_a_row_property() -> None:
    """M0a draws from a seeded generator by position, so it scores the *k*-th
    row rather than a given event — shuffling moves its scores around.

    That is correct for a random floor (every permutation is equally random)
    and it is why `RandomScorer.scores_row_locally` is False. It is recorded
    here because it is the single exception to the rule the test above
    enforces, and an exception nobody wrote down is indistinguishable from a
    bug next time someone reads it.
    """
    frame = _featurized()
    model = RandomScorer()
    model.fit(frame.lazy())

    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=19)

    assert model.score(frame.lazy()).to_list() == model.score(shuffled.lazy()).to_list()
    assert _scores_by_event(model, frame) != _scores_by_event(model, shuffled)
    assert model.scores_row_locally is False


@pytest.mark.parametrize("model", _catalog(), ids=lambda m: str(m.name))
def test_scoring_is_repeatable_for_a_fixed_frame(model: object) -> None:
    """Same model, same frame, twice: NFR-02 at the smallest scale there is."""
    frame = _featurized()
    model.fit(frame.lazy())  # type: ignore[attr-defined]

    first = model.score(frame.lazy()).to_list()  # type: ignore[attr-defined]
    second = model.score(frame.lazy()).to_list()  # type: ignore[attr-defined]

    assert first == second


@pytest.mark.parametrize("model", _catalog(), ids=lambda m: str(m.name))
def test_every_catalog_model_declares_whether_it_is_row_local(model: object) -> None:
    """`scores_row_locally` decides whether `train_eval` may score a model in
    chunks, and getting it wrong produces plausible numbers rather than an
    error. `BaseAnomalyScorer` defaults it to False for exactly that reason —
    it is a safe default, not an answer, so a shipped model must state its own.

    Declaring it on a shared intermediate class counts: `_PyODScorer` sets it
    once for every PyOD-backed model and explains why in the same place, which
    is a decision, not an inheritance accident. Only falling through to
    `BaseAnomalyScorer` fails.
    """
    owners = [
        klass.__name__
        for klass in type(model).__mro__
        if "scores_row_locally" in vars(klass) and klass is not BaseAnomalyScorer
    ]

    assert owners, (
        f"{model.name} falls through to BaseAnomalyScorer's conservative default "  # type: ignore[attr-defined]
        "instead of declaring scores_row_locally. Chunked scoring is a correctness "
        "question, and a default is not an answer to it."
    )
