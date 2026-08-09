"""F1's frequency encoding must be *fitted*, on the training split, once.

Regression guard for a leak that inflated a headline number without breaking
anything visible: `compute_f1` used to derive each category's frequency from
whatever frame it was handed. Since the pipeline featurizes train, val and
test separately, that meant (a) an event's encoding depended on events that
came after it inside its own split, and (b) `auth_type=Kerberos` was a
different number in train than in test, so every model consuming the F1
matrix was scored on a feature whose meaning had shifted underneath it. On
the demo sample this alone moved M2b's AUC-PR by an order of magnitude.
"""

from __future__ import annotations

import polars as pl
import pytest

from authbench.features.event import (
    UNSEEN_CATEGORY_FREQUENCY,
    compute_f1,
    fit_frequency_encoding,
)


def _frame(auth_types: list[str | None]) -> pl.LazyFrame:
    n = len(auth_types)
    return pl.LazyFrame(
        {
            "event_id": list(range(n)),
            "time": [i * 60 for i in range(n)],
            "src_user": ["U1"] * n,
            "dst_user": ["U2"] * n,
            "src_domain": ["D1"] * n,
            "dst_domain": ["D1"] * n,
            "src_computer": ["C1"] * n,
            "dst_computer": ["C2"] * n,
            "auth_type": auth_types,
            "auth_type_is_null": [t is None for t in auth_types],
            "logon_type": ["Network"] * n,
            "logon_type_is_null": [False] * n,
            "auth_orientation": ["LogOn"] * n,
            "success": [True] * n,
            "src_user_is_machine": [False] * n,
        }
    )


def test_same_category_gets_the_same_frequency_in_every_split() -> None:
    # Kerberos is 3/4 of train but only 1/4 of test. Under the old per-frame
    # encoding these produced 0.75 and 0.25 for the identical category.
    train = _frame(["Kerberos", "Kerberos", "Kerberos", "NTLM"])
    test = _frame(["Kerberos", "NTLM", "NTLM", "NTLM"])

    encoding = fit_frequency_encoding(train)
    train_encoded = compute_f1(train, encoding).collect()
    test_encoded = compute_f1(test, encoding).collect()

    assert train_encoded["auth_type_freq"].to_list() == pytest.approx([0.75, 0.75, 0.75, 0.25])
    assert test_encoded["auth_type_freq"].to_list() == pytest.approx([0.75, 0.25, 0.25, 0.25])


def test_a_category_absent_from_training_encodes_as_zero() -> None:
    encoding = fit_frequency_encoding(_frame(["Kerberos", "Kerberos"]))
    encoded = compute_f1(_frame(["Kerberos", "Negotiate"]), encoding).collect()

    assert encoded["auth_type_freq"][1] == pytest.approx(UNSEEN_CATEGORY_FREQUENCY)


def test_null_is_encoded_as_its_own_training_rate_never_as_nan() -> None:
    """~55% of LANL's `auth_type` is null. Nullity is a category here, and the
    encoded column feeds a model matrix directly — a null would land in
    scikit-learn as a NaN rather than as a value.
    """
    encoding = fit_frequency_encoding(_frame(["Kerberos", None, None, None]))
    encoded = compute_f1(_frame([None, "Kerberos"]), encoding).collect()

    assert encoded["auth_type_freq"].null_count() == 0
    assert encoded["auth_type_freq"].to_list() == pytest.approx([0.75, 0.25])


def test_fitting_on_an_empty_training_split_is_an_error_not_a_zero_table() -> None:
    with pytest.raises(ValueError, match="empty training split"):
        fit_frequency_encoding(_frame([]))
