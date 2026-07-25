"""Streaming conversion of raw LANL text logs into day-partitioned ZSTD Parquet (US-102).

Reads the source file in bounded batches — via Polars' batched CSV reader,
which transparently handles gzip — so peak memory stays flat regardless of
file size (NFR-01: no stage exceeds 8 GB RSS). Each batch is cleaned
(`parse.clean.clean_auth`) and written to `out_dir/day=<D>/part-<NNNNNN>.parquet`
immediately, so raw data is read from disk exactly once.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from authbench.parse.clean import DropCounts, clean_auth
from authbench.parse.schema import RAW_AUTH_COLUMNS, RAW_AUTH_SCHEMA

logger = logging.getLogger(__name__)


@dataclass
class ConversionReport:
    n_rows_in: int = 0
    n_rows_out: int = 0
    n_batches: int = 0
    peak_rss_bytes: int = 0
    drop_counts: DropCounts = field(default_factory=DropCounts)


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return 0


def to_parquet_partitioned(
    src_path: Path,
    out_dir: Path,
    *,
    batch_size: int = 2_000_000,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
) -> ConversionReport:
    """Convert `src_path` (plain or gzipped CSV, no header) to a Parquet
    dataset partitioned by `day=<D>` under `out_dir`.

    Memory stays bounded by `batch_size` rows per batch — never the full file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report = ConversionReport()

    reader = pl.read_csv_batched(
        str(src_path),
        has_header=False,
        new_columns=RAW_AUTH_COLUMNS,
        schema_overrides=RAW_AUTH_SCHEMA,
        batch_size=batch_size,
    )

    batch_idx = 0
    id_offset = 0
    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        for raw_batch in batches:
            report.n_rows_in += raw_batch.height
            typed_lazy, batch_drops = clean_auth(raw_batch.lazy(), id_offset=id_offset)
            typed = typed_lazy.collect()
            id_offset += raw_batch.height

            report.drop_counts.malformed_user_domain += batch_drops.malformed_user_domain
            report.drop_counts.null_time += batch_drops.null_time
            report.n_rows_out += typed.height

            for day_value, day_df in typed.partition_by("day", as_dict=True).items():
                day = day_value[0] if isinstance(day_value, tuple) else day_value
                day_dir = out_dir / f"day={day}"
                day_dir.mkdir(parents=True, exist_ok=True)
                day_df.write_parquet(
                    day_dir / f"part-{batch_idx:06d}.parquet",
                    compression=compression,
                )

            batch_idx += 1
            report.n_batches += 1
            report.peak_rss_bytes = max(report.peak_rss_bytes, _current_rss_bytes())

    return report


def verify_row_count(report: ConversionReport, expected_rows: int) -> None:
    """US-102: converted row count must match the published LANL total exactly
    (after accounting for explicitly-counted drops), or the pipeline aborts.
    """
    accounted_for = report.n_rows_out + report.drop_counts.total
    if accounted_for != expected_rows:
        raise ValueError(
            f"Row count mismatch: {accounted_for} accounted for "
            f"({report.n_rows_out} kept + {report.drop_counts.total} dropped), "
            f"expected {expected_rows}. Aborting — see US-102."
        )
