from __future__ import annotations

import polars as pl

from authbench.parse.clean import clean_auth, clean_redteam
from authbench.parse.schema import RAW_AUTH_COLUMNS, RAW_REDTEAM_COLUMNS


def _raw_auth_frame(rows: list[list[object]]) -> pl.LazyFrame:
    return pl.LazyFrame(rows, schema=RAW_AUTH_COLUMNS, orient="row").lazy()


def test_machine_account_flagged_not_dropped() -> None:
    rows = [
        [100, "alice@DOM", "alice@DOM", "C1", "C2", "Kerberos", "Network", "LogOn", "Success"],
        [200, "PC01$@DOM", "PC01$@DOM", "C1", "C2", "?", "?", "LogOn", "Success"],
    ]
    typed, counts = clean_auth(_raw_auth_frame(rows))
    result = typed.collect()

    assert counts.total == 0
    assert result.height == 2
    assert (
        result.filter(pl.col("src_user").cast(pl.Utf8) == "alice")["src_user_is_machine"][0]
        is False
    )
    assert (
        result.filter(pl.col("src_user").cast(pl.Utf8) == "PC01$")["src_user_is_machine"][0] is True
    )


def test_null_token_becomes_indicator_not_imputation() -> None:
    rows = [
        [100, "alice@DOM", "alice@DOM", "C1", "C2", "?", "Network", "LogOn", "Success"],
    ]
    typed, _ = clean_auth(_raw_auth_frame(rows))
    result = typed.collect()

    assert result["auth_type_is_null"][0] is True
    assert result["auth_type"][0] is None
    assert result["logon_type_is_null"][0] is False


def test_malformed_user_domain_dropped_and_counted() -> None:
    rows = [
        [100, "alice@DOM", "alice@DOM", "C1", "C2", "Kerberos", "Network", "LogOn", "Success"],
        [200, "no-domain-here", "alice@DOM", "C1", "C2", "Kerberos", "Network", "LogOn", "Success"],
    ]
    typed, counts = clean_auth(_raw_auth_frame(rows))
    result = typed.collect()

    assert result.height == 1
    assert counts.malformed_user_domain == 1
    assert counts.total == 1


def test_event_id_unique_across_batches_via_offset() -> None:
    rows = [
        [100, "alice@DOM", "alice@DOM", "C1", "C2", "Kerberos", "Network", "LogOn", "Success"],
    ]
    batch0, _ = clean_auth(_raw_auth_frame(rows), id_offset=0)
    batch1, _ = clean_auth(_raw_auth_frame(rows), id_offset=1)

    id0 = batch0.collect()["event_id"][0]
    id1 = batch1.collect()["event_id"][0]
    assert id0 != id1


def test_clean_redteam_deduplicates() -> None:
    raw = pl.LazyFrame(
        [
            [100, "alice@DOM", "C1", "C2"],
            [100, "alice@DOM", "C1", "C2"],  # exact duplicate
        ],
        schema=RAW_REDTEAM_COLUMNS,
        orient="row",
    ).lazy()
    typed, n_dupes = clean_redteam(raw)
    assert typed.collect().height == 1
    assert n_dupes == 1
