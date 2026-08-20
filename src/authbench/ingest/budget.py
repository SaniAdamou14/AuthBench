"""What a full-dataset run costs, in bytes and in RAM, before it is started.

Every constant here is measured on the demo sample and scaled by event count,
not guessed: `authbench demo` writes a 90-column ZSTD feature store, and the
per-event sizes below come straight off it. The one genuinely unknown factor
is LANL's own gzip ratio, which nobody can compute without the file — so the
download line is an assumption, replaced by the server's `Content-Length` the
moment `authbench data download` runs.

The point is not precision. It is that a 1.05-billion-event run should fail on
a printed number before the download starts, not with `No space left on
device` five hours into a Parquet conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from authbench.ingest.download import free_space_bytes

# --- Measured on the demo sample (37,826 events, data/demo/auth_demo.txt) ----

#: Raw LANL-shaped text, bytes per auth event.
RAW_TEXT_BYTES_PER_EVENT = 68.7

#: Day-partitioned ZSTD Parquet of the typed 17-column schema, bytes per event.
INTERIM_PARQUET_BYTES_PER_EVENT = 11.3

#: ZSTD Parquet feature store, F1–F4, 90 columns, bytes per event.
FEATURE_STORE_BYTES_PER_EVENT = 94.1

#: Columns in `features.MODEL_FEATURE_COLUMNS`, as float64, in RAM.
MODEL_MATRIX_BYTES_PER_EVENT = 21 * 8

# --- Assumptions, stated rather than hidden ---------------------------------

#: LANL's auth.txt.gz against its own decompressed size. Highly repetitive
#: text; refined from the server's Content-Length at download time.
ASSUMED_GZIP_RATIO = 0.08

#: The demo sample has fewer distinct users and machines than LANL's 12,425
#: and 17,684, and dictionary-encoded Parquet compresses better the fewer
#: distinct values it holds. Every measured per-event size above is inflated
#: by this factor before it is reported, so the estimate errs high.
CARDINALITY_PENALTY = 1.4

#: PyOD/sklearn estimators copy the design matrix (a scaled copy, then a
#: transformed one). Peak is a small multiple of the matrix itself.
ESTIMATOR_COPY_FACTOR = 3.0

LANL_TOTAL_EVENTS = 1_051_430_459
LANL_TRAIN_FRACTION = 30 / 58
LANL_TEST_FRACTION = 18 / 58


@dataclass(frozen=True)
class StageBudget:
    """What one pipeline stage adds to disk and needs in memory."""

    stage: str
    disk_bytes: int
    peak_rss_bytes: int
    note: str


def lanl_budget(
    n_events: int = LANL_TOTAL_EVENTS,
    *,
    train_fraction: float = LANL_TRAIN_FRACTION,
    test_fraction: float = LANL_TEST_FRACTION,
) -> list[StageBudget]:
    """Per-stage disk and memory budget for a run over `n_events` auth events.

    `n_events` is a parameter rather than a constant so the same table answers
    "what would a 10-day slice cost?" — which is the question that matters
    once the full figure turns out not to fit.
    """
    penalty = CARDINALITY_PENALTY
    raw_text = n_events * RAW_TEXT_BYTES_PER_EVENT
    n_train = int(n_events * train_fraction)
    n_test = int(n_events * test_fraction)

    return [
        StageBudget(
            stage="download (auth.txt.gz + redteam.txt.gz)",
            disk_bytes=int(raw_text * ASSUMED_GZIP_RATIO),
            peak_rss_bytes=64 * 1024**2,
            note="streamed to disk; memory is one HTTP chunk",
        ),
        StageBudget(
            stage="to_parquet (data/interim/auth)",
            disk_bytes=int(n_events * INTERIM_PARQUET_BYTES_PER_EVENT * penalty),
            peak_rss_bytes=2 * 1024**3,
            note="bounded by --block-bytes, not by file size",
        ),
        StageBudget(
            stage="label_split_features (data/processed/features)",
            disk_bytes=int(n_events * FEATURE_STORE_BYTES_PER_EVENT * penalty),
            peak_rss_bytes=8 * 1024**3,
            note="F2 diversity self-joins are NOT streamable at this scale",
        ),
        StageBudget(
            stage="train_eval (design matrix in RAM)",
            disk_bytes=64 * 1024**2,
            peak_rss_bytes=int(
                (n_train + n_test) * MODEL_MATRIX_BYTES_PER_EVENT * ESTIMATOR_COPY_FACTOR
            ),
            note="pl.read_parquet loads each split whole; PCA/ECOD copy it",
        ),
    ]


def total_disk_bytes(budget: list[StageBudget]) -> int:
    """Peak disk, i.e. everything at once.

    No stage deletes its predecessor's output — DVC needs the inputs of every
    stage it might re-run — so the three data directories coexist.
    """
    return sum(stage.disk_bytes for stage in budget)


def peak_rss_bytes(budget: list[StageBudget]) -> int:
    """Peak memory: stages run one at a time, so this is the largest, not the sum."""
    return max(stage.peak_rss_bytes for stage in budget)


def max_events_for_disk(available_bytes: int, *, margin_bytes: int = 5 * 1024**3) -> int:
    """Largest `n_events` whose full budget fits in `available_bytes`.

    Inverts `lanl_budget`'s per-event disk terms; the answer is what makes a
    "the full dataset does not fit" message actionable instead of merely true.
    """
    per_event = (
        RAW_TEXT_BYTES_PER_EVENT * ASSUMED_GZIP_RATIO
        + (INTERIM_PARQUET_BYTES_PER_EVENT + FEATURE_STORE_BYTES_PER_EVENT) * CARDINALITY_PENALTY
    )
    usable = max(0, available_bytes - margin_bytes)
    return int(usable / per_event)


def max_events_for_memory(
    available_bytes: int,
    *,
    train_fraction: float = LANL_TRAIN_FRACTION,
    test_fraction: float = LANL_TEST_FRACTION,
) -> int:
    """Largest `n_events` whose train+test design matrix fits in `available_bytes`."""
    per_event = (
        (train_fraction + test_fraction) * MODEL_MATRIX_BYTES_PER_EVENT * ESTIMATOR_COPY_FACTOR
    )
    return int(available_bytes / per_event)


def available_memory_bytes() -> int:
    """Physical RAM, or 0 if psutil is unavailable."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return 0
    return int(psutil.virtual_memory().total)


def available_disk_bytes(path: Path) -> int:
    return free_space_bytes(path)
