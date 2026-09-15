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
    # Added alongside the wider-window LANL split: a value here that disagreed
    # with conf/ would be the exact drift this test file exists to catch, and
    # this key controls how many extra days each split scans before its own —
    # a silent mismatch changes how much history a run actually got without
    # changing what `dvc params diff` claims it asked for.
    assert params.features.history_warmup_days == conf.history_warmup_days  # type: ignore[union-attr]


def test_the_split_lies_inside_the_converted_day_window() -> None:
    """`to_parquet_days` decides which days exist on disk; the split decides
    which days the pipeline asks for. A split reaching outside the window does
    not produce an empty partition, it produces a *missing* one — and
    `verify_temporal_order` then reports a leakage error for a reason that has
    nothing to do with leakage."""
    params = _load("params.yaml")
    window = str(params.to_parquet_days).strip()  # type: ignore[union-attr]
    if not window:
        return  # every day converted; nothing to check

    first, last = (int(part) for part in window.split(":"))
    conf = _load("conf", "split", "temporal.yaml")
    for key in ("train_days", "val_days", "test_days"):
        low, high = (int(v) for v in conf[key])  # type: ignore[index]
        assert first <= low <= high <= last, (
            f"split.{key} = [{low}, {high}] falls outside the converted window {window}"
        )


def test_runtime_knobs_match() -> None:
    params = _load("params.yaml")
    conf = _load("conf", "config.yaml")

    assert params.runtime.fit_sample_size == conf.runtime.fit_sample_size  # type: ignore[union-attr]
    assert params.runtime.bootstrap_workers == conf.runtime.bootstrap_workers  # type: ignore[union-attr]


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


def test_the_demo_resamples_can_resolve_a_significant_result_too() -> None:
    """The same arithmetic, on the side of it nothing was watching.

    `authbench demo` pins its own `DEMO_BOOTSTRAP_RESAMPLES` instead of reading
    `conf/eval/default.yaml`, so the guard above never covered it. Its 7-model
    catalog needs 839 resamples and it has 1000 — but adding a single model
    (HBOS, say, which the full catalog already carries) takes it to 28 pairs
    and 1119, and every demo comparison would silently come back
    non-significant while reading like a real null result. That is the exact
    failure `conf/` is guarded against; the demo now is too.
    """
    from authbench.cli import DEMO_BOOTSTRAP_RESAMPLES, demo_model_catalog
    from authbench.evaluate.stats_tests import minimum_resamples_for_family

    n_models = len(demo_model_catalog())
    n_pairs = n_models * (n_models - 1) // 2
    alpha = float(_load("conf", "eval", "default.yaml").pairwise_test.alpha)  # type: ignore[union-attr]

    required = minimum_resamples_for_family(n_pairs, alpha)
    assert required <= DEMO_BOOTSTRAP_RESAMPLES, (
        f"the demo's {n_models} models = {n_pairs} pairs needs >= {required} resamples, "
        f"DEMO_BOOTSTRAP_RESAMPLES is {DEMO_BOOTSTRAP_RESAMPLES}"
    )
