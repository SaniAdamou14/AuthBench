"""`params.yaml` mirrors `conf/`, and drift between them is a build failure.

DVC reads `params.yaml`; the pipeline stages read `conf/` through Hydra. A
value that disagrees between the two does not change a single number in the
run — it changes what `dvc params diff` claims the run was. That is worse
than a wrong value, because it is a wrong value that looks authoritative, and
it had already happened once: `bootstrap_resamples` sat at 1000 in
`params.yaml` long after `conf/eval/default.yaml` moved to 2000, the exact
figure the README explains cannot produce a significant result.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(*parts: str) -> object:
    return OmegaConf.load(REPO_ROOT.joinpath(*parts))


def test_split_bounds_match() -> None:
    params = _load("params.yaml")
    conf = _load("conf", "split", "temporal.yaml")

    for key in ("train_days", "val_days", "test_days"):
        assert list(params.split[key]) == list(conf[key]), f"split.{key} drifted"  # type: ignore[index,union-attr]


def test_evaluation_knobs_match() -> None:
    params = _load("params.yaml")
    conf = _load("conf", "eval", "default.yaml")

    assert list(params.eval.budgets) == list(conf.budgets)  # type: ignore[union-attr]
    assert list(params.eval.fpr_targets) == list(conf.fpr_targets)  # type: ignore[union-attr]
    assert params.eval.bootstrap_resamples == conf.bootstrap.n_resamples  # type: ignore[union-attr]
    assert params.eval.bootstrap_confidence == conf.bootstrap.confidence  # type: ignore[union-attr]
    assert params.eval.pairwise_alpha == conf.pairwise_test.alpha  # type: ignore[union-attr]


def test_feature_knobs_match() -> None:
    params = _load("params.yaml")
    conf = _load("conf", "features", "base.yaml")

    assert list(params.features.f2_windows_hours) == list(conf.f2_history.windows_hours)  # type: ignore[union-attr]
    assert params.features.f5_graph_enabled == conf.f5_graph.enabled  # type: ignore[union-attr]
    assert params.features.f6_sequence_enabled == conf.f6_sequence.enabled  # type: ignore[union-attr]
    assert params.features.f6_context_length == conf.f6_sequence.context_length  # type: ignore[union-attr]


def test_dataset_knobs_match() -> None:
    params = _load("params.yaml")
    conf = _load("conf", "dataset", f"{_load('params.yaml').dataset}.yaml")  # type: ignore[union-attr]

    assert params.campaign_gap_hours == conf.campaign_gap_hours  # type: ignore[union-attr]


def test_the_configured_resamples_can_resolve_a_significant_result() -> None:
    """The arithmetic the README describes, asserted rather than described.

    With the 8-model catalog Holm's strictest threshold is alpha/28, and a
    percentile bootstrap cannot report below 2/(R+1). If the config ever goes
    back below that bound, every comparison comes back non-significant by
    arithmetic and reads exactly like a real null result.
    """
    from authbench.evaluate.stats_tests import minimum_resamples_for_family
    from authbench.pipeline.train_eval import build_model_catalog

    n_models = len(build_model_catalog())
    n_pairs = n_models * (n_models - 1) // 2
    conf = _load("conf", "eval", "default.yaml")

    required = minimum_resamples_for_family(n_pairs, float(conf.pairwise_test.alpha))  # type: ignore[union-attr]
    assert int(conf.bootstrap.n_resamples) >= required, (  # type: ignore[union-attr]
        f"{n_models} models = {n_pairs} pairs needs >= {required} resamples"
    )
