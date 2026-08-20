"""The demo and the DVC stage must fit the *same* models.

`authbench demo` and `pipeline.train_eval` each build a catalog containing
`M2b_pca_reconstruction`, `M3a_iforest` and `M3b_ecod`. If they hand those
estimators different design matrices, the two produce different models under
identical names, and the demo stops being a smoke test of the real pipeline.
It had already drifted: the DVC stage carried its own 14-column list, missing
every F2 history feature and all three F1 frequency encodings.
"""

from __future__ import annotations

import polars as pl

from authbench.features import MODEL_FEATURE_COLUMNS


def test_the_design_matrix_has_a_single_definition() -> None:
    import authbench.cli as cli
    import authbench.pipeline.train_eval as train_eval

    assert cli.MODEL_FEATURE_COLUMNS is MODEL_FEATURE_COLUMNS
    assert train_eval.MODEL_FEATURE_COLUMNS is MODEL_FEATURE_COLUMNS


def test_every_model_in_both_catalogs_uses_it() -> None:
    from authbench.models.classical import _PyODScorer
    from authbench.models.stats import PCAReconstructionScorer
    from authbench.pipeline.train_eval import build_model_catalog

    matrix_models = [
        model
        for model in build_model_catalog()
        if isinstance(model, _PyODScorer | PCAReconstructionScorer)
    ]

    assert matrix_models, "the catalog should contain vector-space models"
    for model in matrix_models:
        assert model.feature_cols == MODEL_FEATURE_COLUMNS


def test_no_column_carries_the_infinite_sentinel() -> None:
    """`days_since_pair_last_seen` and `hours_since_prev_event_same_user` use
    +inf as a documented "never seen before" sentinel (US-111). A standardizer
    turns that into NaN across the whole column, so they stay out of the matrix
    — `pair_is_new` already carries the same information as a boolean.
    """
    assert "days_since_pair_last_seen" not in MODEL_FEATURE_COLUMNS
    assert "hours_since_prev_event_same_user" not in MODEL_FEATURE_COLUMNS


def test_the_columns_exist_on_a_fully_featurized_frame() -> None:
    """The list is only useful if the feature pipeline actually produces it."""
    from authbench.features.event import compute_f1, fit_frequency_encoding
    from authbench.features.history import compute_f2
    from authbench.features.novelty import compute_f3
    from authbench.features.temporal import calibrate_night_window, compute_f4

    events = pl.LazyFrame(
        {
            "event_id": list(range(6)),
            "time": [0, 60, 3600, 90_000, 90_060, 180_000],
            "day": [0, 0, 0, 1, 1, 2],
            "src_user": ["U1", "U1", "U2", "U1", "U2", "U1"],
            "dst_user": ["U1", "U1", "U2", "U1", "U2", "U1"],
            "src_domain": ["D1"] * 6,
            "dst_domain": ["D1"] * 6,
            "src_computer": ["C1", "C1", "C2", "C1", "C2", "C3"],
            "dst_computer": ["C2", "C3", "C2", "C4", "C5", "C6"],
            "auth_type": ["Negotiate", "Kerberos", None, "Negotiate", "NTLM", "Kerberos"],
            "auth_type_is_null": [False, False, True, False, False, False],
            "logon_type": ["Batch", "Network", "Batch", None, "Batch", "Network"],
            "logon_type_is_null": [False, False, False, True, False, False],
            "auth_orientation": ["LogOn"] * 6,
            "success": [True, False, True, True, False, True],
            "src_user_is_machine": [False] * 6,
            "dst_user_is_machine": [False] * 6,
        }
    )

    encoding = fit_frequency_encoding(events)
    night_window = calibrate_night_window(events)
    featurized = compute_f4(compute_f3(compute_f2(compute_f1(events, encoding))), night_window)

    missing = set(MODEL_FEATURE_COLUMNS) - set(featurized.collect_schema().names())
    assert not missing, f"feature pipeline does not produce: {sorted(missing)}"
