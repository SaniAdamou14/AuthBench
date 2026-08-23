"""Streaming conversion of raw LANL text logs into day-partitioned ZSTD Parquet (US-102).

The source is read as a sequence of **bounded byte blocks, each cut on a line
boundary**, decompressed on the fly by `gzip` when needed. Peak memory is then
a property of `block_bytes`, not of the file: `auth.txt.gz` is ~5 GB
compressed and ~65 GB expanded, and any reader that materializes the
decompressed text before parsing it is unusable on a machine with 16 GB of
RAM. Polars' own CSV readers decide that for us and are not contractually
bounded here, so the blocking is done explicitly and the guarantee is ours
(NFR-01: no stage exceeds 8 GB RSS).

Each block is parsed, cleaned (`parse.clean.clean_auth`) and written to
`out_dir/day=<D>/part-<NNNNNN>.parquet` immediately, so the raw file is read
exactly once and never held in full.
"""

from __future__ import annotations

import gzip
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal, cast

import polars as pl

from authbench.ingest.download import free_space_bytes, require_free_space
from authbench.parse.clean import DropCounts, clean_auth
from authbench.parse.schema import RAW_AUTH_COLUMNS

logger = logging.getLogger(__name__)

# ~128 MiB of decompressed text per block: about 2M LANL auth rows, which is
# the batch size this stage has always used, expressed in the unit that
# actually bounds memory.
DEFAULT_BLOCK_BYTES = 128 * 1024 * 1024

# ZSTD Parquet of the typed schema against raw *uncompressed* text, measured
# on the demo sample and consistent with the published LANL row/byte counts.
# Used only to warn about disk before a multi-hour conversion, never to make a
# decision about the data.
PARQUET_TO_RAW_TEXT_RATIO = 0.22

_READ_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass
class ConversionReport:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_batches: int = 0
    peak_rss_bytes: int = 0
    drop_counts: DropCounts = field(default_factory=DropCounts)
    # Set when the conversion was restricted to a window of days. Rows outside
    # it are *excluded*, not *dropped*: nothing was wrong with them, so they
    # are counted separately and never folded into `drop_counts`, which exists
    # to account for data the pipeline could not use.
    day_range: tuple[int, int] | None = None
    n_rows_out_of_range: int = 0
    stopped_early: bool = False

    @property
    def is_partial(self) -> bool:
        return self.day_range is not None


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return 0


def _open_maybe_gzip(src_path: Path) -> IO[bytes]:
    """Open `src_path` for binary reading, transparently decompressing gzip.

    Detected by magic bytes rather than by suffix: LANL's files happen to be
    named `.txt.gz`, but a file someone decompressed and kept the name of is a
    far more common accident than the reverse.
    """
    with src_path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return cast("IO[bytes]", gzip.open(src_path, "rb"))
    return src_path.open("rb")


def iter_line_blocks(src_path: Path, block_bytes: int = DEFAULT_BLOCK_BYTES) -> Iterator[bytes]:
    """Yield the file's bytes in blocks of at most ~`block_bytes`, each ending
    on a newline so no record is ever split across two blocks.

    This is the whole memory guarantee of this module: at most one block, plus
    the tail of the current line, is resident at a time.
    """
    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be positive, got {block_bytes}.")

    read_size = min(_READ_CHUNK_BYTES, block_bytes)
    buffer = bytearray()
    with _open_maybe_gzip(src_path) as handle:
        while True:
            chunk = handle.read(read_size)
            if not chunk:
                break
            buffer += chunk
            while len(buffer) >= block_bytes:
                cut = buffer.rfind(b"\n", 0, block_bytes)
                if cut == -1:
                    # A single line longer than block_bytes: run past the
                    # limit to the end of that line rather than emit a
                    # truncated record. If it hasn't arrived yet, read more.
                    cut = buffer.find(b"\n", block_bytes)
                    if cut == -1:
                        break
                yield bytes(buffer[: cut + 1])
                del buffer[: cut + 1]

    if buffer.strip():
        yield bytes(buffer)


def _parse_block(block: bytes) -> pl.DataFrame:
    """Parse one raw block into the 9 raw auth columns.

    Every field is read as text and `time` is cast non-strictly afterwards. A
    strict integer schema would abort the whole multi-hour conversion on a
    single malformed timestamp in 1.05 billion rows; a non-strict cast turns
    it into a null, which `clean_auth` already drops *and counts* under
    `null_time` — the reconciliation US-102 checks stays exact either way.
    """
    frame = pl.read_csv(
        block,
        has_header=False,
        new_columns=RAW_AUTH_COLUMNS,
        schema_overrides=dict.fromkeys(RAW_AUTH_COLUMNS, pl.Utf8),
        truncate_ragged_lines=True,
    )
    # A block containing a row with extra fields gains a `column_10`; keeping
    # it would give that one block a different schema from every other one.
    # Projecting to the nine documented columns makes every block's schema
    # identical by construction, which is what lets them land in one dataset.
    return frame.select(RAW_AUTH_COLUMNS).with_columns(pl.col("time").cast(pl.Int64, strict=False))


def estimate_parquet_bytes(
    src_path: Path, *, days: tuple[int, int] | None = None, total_days: int = 58
) -> int:
    """Rough expected size of the Parquet dataset for `src_path`.

    Gzip's own compression ratio is unknown without decompressing, so a
    conservative 10x expansion is assumed for a `.gz` source. `days` scales the
    estimate by the fraction of the period being converted, assuming an even
    spread of events across days. Used for the disk-space warning only.
    """
    compressed_bytes = src_path.stat().st_size
    with src_path.open("rb") as probe:
        is_gzip = probe.read(2) == b"\x1f\x8b"
    raw_text_bytes = compressed_bytes * 10 if is_gzip else compressed_bytes
    fraction = 1.0 if days is None else min(1.0, (days[1] - days[0] + 1) / max(1, total_days))
    return int(raw_text_bytes * PARQUET_TO_RAW_TEXT_RATIO * fraction)


def to_parquet_partitioned(
    src_path: Path,
    out_dir: Path,
    *,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    check_disk_space: bool = True,
    days: tuple[int, int] | None = None,
) -> ConversionReport:
    """Convert `src_path` (plain or gzipped CSV, no header) to a Parquet
    dataset partitioned by `day=<D>` under `out_dir`.

    Memory stays bounded by `block_bytes` — never the full file.

    `days=(first, last)` restricts the output to that inclusive window of
    `time // 86400`. That is what makes a real-dataset run possible on a
    machine that cannot hold all 58 days of LANL: two weeks costs roughly a
    quarter of the disk and, because `auth.txt` is written in time order, only
    about a quarter of the reading. Rows outside the window are counted in
    `n_rows_out_of_range`, never in `drop_counts` — they are excluded, not
    rejected, and conflating the two would corrupt US-102's reconciliation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if days is not None and days[0] > days[1]:
        raise ValueError(f"days must be (first, last) with first <= last, got {days}.")

    if check_disk_space:
        # Warn-then-check: the estimate is coarse, so it is reported either
        # way and only refuses when it is not close.
        estimated = estimate_parquet_bytes(src_path, days=days)
        logger.info(
            "Estimated Parquet output: ~%.1f GB (%.1f GB free on the destination volume).",
            estimated / 1e9,
            free_space_bytes(out_dir) / 1e9,
        )
        require_free_space(out_dir, estimated, what=f"converting {src_path.name}")

    report = ConversionReport(day_range=days)
    id_offset = 0
    highest_day_seen = -1
    source_is_time_ordered = True

    for batch_idx, block in enumerate(iter_line_blocks(src_path, block_bytes)):
        raw_batch = _parse_block(block)
        report.n_rows_in += raw_batch.height

        typed_lazy, batch_drops = clean_auth(raw_batch.lazy(), id_offset=id_offset)
        typed = typed_lazy.collect()
        # Advanced by the *raw* height, before any filtering, so `event_id`
        # stays a stable identity for a row of the source file no matter which
        # day window this run happens to convert.
        id_offset += raw_batch.height

        report.drop_counts.malformed_user_domain += batch_drops.malformed_user_domain
        report.drop_counts.null_time += batch_drops.null_time

        if days is not None and typed.height:
            block_min_day = int(typed["day"].min())  # type: ignore[arg-type]
            block_max_day = int(typed["day"].max())  # type: ignore[arg-type]
            if block_max_day < highest_day_seen:
                source_is_time_ordered = False
            highest_day_seen = max(highest_day_seen, block_max_day)

            in_range = typed.filter(pl.col("day").is_between(days[0], days[1]))
            report.n_rows_out_of_range += typed.height - in_range.height
            typed = in_range

            # `auth.txt` is written in time order, so once a block starts past
            # the window there is nothing left to find and the remaining
            # gigabytes need not be decompressed. Only taken while the source
            # has actually been monotonic — on an out-of-order file this would
            # silently truncate the output, which is worse than being slow.
            if source_is_time_ordered and block_min_day > days[1]:
                report.stopped_early = True
                logger.info(
                    "Reached day %d, past the requested window %s — stopping early.",
                    block_min_day,
                    days,
                )
                report.n_batches += 1
                break

        report.n_rows_out += typed.height

        for day_value, day_df in typed.partition_by("day", as_dict=True).items():
            day = day_value[0] if isinstance(day_value, tuple) else day_value
            day_dir = out_dir / f"day={day}"
            day_dir.mkdir(parents=True, exist_ok=True)
            day_df.write_parquet(
                day_dir / f"part-{batch_idx:06d}.parquet",
                compression=compression,
            )

        report.n_batches += 1
        report.peak_rss_bytes = max(report.peak_rss_bytes, _current_rss_bytes())

        if report.n_batches % 50 == 0:
            logger.info(
                "%d blocks, %d rows converted, peak RSS %.2f GB",
                report.n_batches,
                report.n_rows_out,
                report.peak_rss_bytes / 1e9,
            )

    return report


class PartialConversionError(ValueError):
    """Raised when a day-restricted conversion is asked to reconcile against
    the whole dataset's published row count — a check it cannot pass and must
    not appear to pass."""


def verify_row_count(report: ConversionReport, expected_rows: int) -> None:
    """US-102: converted row count must match the published LANL total exactly
    (after accounting for explicitly-counted drops), or the pipeline aborts.

    A day-restricted run is refused rather than reconciled. It read part of the
    file and, if it stopped early, never saw the rest — so there is no honest
    total to compare against, and quietly comparing the part to the whole would
    turn the project's strongest data-integrity check into a guaranteed
    failure that everyone learns to ignore.
    """
    if report.is_partial:
        raise PartialConversionError(
            f"This conversion was restricted to days {report.day_range}: "
            f"{report.n_rows_out:,} rows kept, {report.n_rows_out_of_range:,} outside the "
            f"window, and the read {'stopped early' if report.stopped_early else 'ran to the end'}. "
            f"It cannot be reconciled against the full-dataset total of {expected_rows:,}. "
            "Drop --expected-rows (or pass 0) for partial conversions, and label every "
            "downstream number with the day window it came from."
        )

    accounted_for = report.n_rows_out + report.drop_counts.total
    if accounted_for != expected_rows:
        raise ValueError(
            f"Row count mismatch: {accounted_for} accounted for "
            f"({report.n_rows_out} kept + {report.drop_counts.total} dropped), "
            f"expected {expected_rows}. Aborting — see US-102."
        )
