"""US-102: the conversion that will run once, for hours, on 1.05 billion rows.

Until this file existed the module had zero test coverage — the stage that
touches the most data was the least verified thing in the project.
"""

from __future__ import annotations

import gzip
import types
from pathlib import Path

import polars as pl
import pytest

from authbench.ingest.download import InsufficientDiskSpaceError
from authbench.ingest.to_parquet import iter_line_blocks, to_parquet_partitioned, verify_row_count

ROWS = [
    "1,U1@DOM1,U1@DOM1,C1,C2,Negotiate,Batch,LogOn,Success",
    "2,U2@DOM1,U2@DOM1,C3,C4,?,?,LogOn,Fail",
    "86401,C5$@DOM1,U3@DOM2,C5,C6,Kerberos,Network,LogOn,Success",
    "172801,U4@DOM1,U4@DOM1,C7,C7,NTLM,Interactive,LogOff,Success",
]


def _write(directory: Path, lines: list[str], *, gzipped: bool = False) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(lines) + "\n").encode()
    path = directory / ("auth.txt.gz" if gzipped else "auth.txt")
    path.write_bytes(gzip.compress(payload) if gzipped else payload)
    return path


def test_line_blocks_reassemble_into_the_original_file_byte_for_byte(tmp_path: Path) -> None:
    src = _write(tmp_path / "src", ROWS * 200)
    blocks = list(iter_line_blocks(src, block_bytes=1024))

    assert len(blocks) > 1, "block_bytes must actually split the file"
    assert b"".join(blocks) == src.read_bytes()
    assert all(block.endswith(b"\n") for block in blocks), "a record was split across blocks"


def test_a_line_longer_than_a_block_is_never_truncated(tmp_path: Path) -> None:
    long_row = "9," + "U" * 5000 + "@DOM1,U1@DOM1,C1,C2,Negotiate,Batch,LogOn,Success"
    src = _write(tmp_path / "src", [ROWS[0], long_row, ROWS[1]])

    blocks = list(iter_line_blocks(src, block_bytes=64))

    assert b"".join(blocks) == src.read_bytes()
    assert any(long_row.encode() in block for block in blocks)


def test_memory_is_bounded_by_block_size_not_file_size(tmp_path: Path) -> None:
    """The property the LANL run depends on: `auth.txt.gz` expands to roughly
    65 GB, and no resident buffer may be a function of that."""
    src = _write(tmp_path / "src", ROWS * 5000, gzipped=True)

    sizes = [len(block) for block in iter_line_blocks(src, block_bytes=4096)]

    assert max(sizes) < 4096 + 512, "a block grew past block_bytes plus one line"
    assert len(sizes) > 10


@pytest.mark.parametrize("gzipped", [False, True])
def test_gzip_and_plain_sources_produce_identical_datasets(tmp_path: Path, gzipped: bool) -> None:
    src = _write(tmp_path / f"src-{gzipped}", ROWS * 50, gzipped=gzipped)
    out = tmp_path / f"parquet-{gzipped}"

    report = to_parquet_partitioned(src, out, block_bytes=512)

    frame = pl.read_parquet(out / "**" / "*.parquet")
    assert report.n_batches > 1
    assert report.n_rows_in == report.n_rows_out == len(ROWS) * 50 == frame.height
    assert frame["event_id"].n_unique() == frame.height, "event_id must be globally unique"
    assert sorted(frame["day"].unique().to_list()) == [0, 1, 2]
    verify_row_count(report, len(ROWS) * 50)


def test_a_malformed_timestamp_is_counted_not_fatal(tmp_path: Path) -> None:
    """One bad row in 1.05 billion must not abort a multi-hour conversion, and
    must not vanish either — US-102's reconciliation has to stay exact."""
    lines = [*ROWS, "not-a-timestamp,U9@DOM1,U9@DOM1,C1,C2,Negotiate,Batch,LogOn,Success"]
    src = _write(tmp_path / "bad", lines)

    report = to_parquet_partitioned(src, tmp_path / "out", block_bytes=4096)

    assert report.n_rows_in == len(lines)
    assert report.n_rows_out == len(ROWS)
    assert report.drop_counts.null_time == 1
    assert report.drop_counts.malformed_user_domain == 0
    verify_row_count(report, len(lines))


def test_row_count_mismatch_aborts(tmp_path: Path) -> None:
    src = _write(tmp_path / "count", ROWS)
    report = to_parquet_partitioned(src, tmp_path / "out", block_bytes=4096)

    with pytest.raises(ValueError, match="Row count mismatch"):
        verify_row_count(report, len(ROWS) + 1)


def test_conversion_refuses_to_start_without_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _write(tmp_path / "disk", ROWS * 100)
    monkeypatch.setattr(
        "authbench.ingest.download.shutil.disk_usage",
        lambda _: types.SimpleNamespace(total=1024, used=0, free=1024),
    )

    with pytest.raises(InsufficientDiskSpaceError, match="GB"):
        to_parquet_partitioned(src, tmp_path / "out", block_bytes=512)
