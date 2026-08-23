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
#:
#: 6.9, so that 6.9 x 1.4 = 9.7 — the figure **measured on LANL itself**
#: (2.31 GB for the 239,471,459 events of days 0-13), not extrapolated from the
#: demo. The demo-derived 11.3 was 60% too high.
INTERIM_PARQUET_BYTES_PER_EVENT = 6.9

#: ZSTD Parquet feature store, F1–F4, bytes per event.
#:
#: The store persists `FEATURE_STORE_COLUMNS` (33 columns) instead of
#: everything the feature pipeline can produce (90); the dropped 57 were read
#: by nothing. The same change removed eight of the nine `causal_prior_events`
#: self-joins, and the ninth became an interval sweep — on LANL day 0 that one
#: would have built 34.6 billion pairs and instead takes 9.7 seconds.
#:
#: Now 25.4, so that 25.4 x 1.4 = 35.5 — again **measured on LANL**, over a
#: three-day feature build of 48.9 million real events. Higher cardinality did
#: not cost what the demo predicted: dictionary-encoded Parquet handles 12,425
#: users and 17,684 machines better than the penalty assumed.
FEATURE_STORE_BYTES_PER_EVENT = 25.4

#: Columns in `features.MODEL_FEATURE_COLUMNS`, as float64, in RAM.
MODEL_MATRIX_BYTES_PER_EVENT = 21 * 8

#: The evaluation frame the bootstrap runs on, per event: `EVAL_COLUMNS`
#: (~23 bytes), one float64 score per model in the catalog (8 x 8), the single
#: pre-sorted rank order held at a time (int32 permutation + uint8 label mask +
#: int32 group boundaries, ~9), and the int32 resample count vector (4).
#:
#: The rank orders used to be held for all eight models at once, as int64 and
#: float64 — 128 bytes per event instead of 9. Building them one at a time,
#: each replaying the same seeded draw sequence, keeps the resamples paired and
#: drops this line by more than half.
SCORED_FRAME_BYTES_PER_EVENT = 23 + 8 * 8 + 9 + 4

# --- Assumptions, stated rather than hidden ---------------------------------

#: LANL's auth.txt.gz against its own decompressed size.
#:
#: 0.106, **measured**: the real file is 7,626,505,158 bytes against an
#: estimated 72.2 GB of raw text. The 0.08 this started as was a guess, and it
#: understated the download by 1.8 GB — which on a laptop with single-digit
#: gigabytes free is the difference between a plan that works and one that
#: dies after five hours of transfer.
ASSUMED_GZIP_RATIO = 0.106

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
    test_fraction: float = LANL_TEST_FRACTION,
    fit_sample_size: int = 5_000_000,
    scoring_chunk_rows: int = 2_000_000,
) -> list[StageBudget]:
    """Per-stage disk and memory budget for a run over `n_events` auth events.

    `n_events` is a parameter rather than a constant so the same table answers
    "what would a 10-day slice cost?" — which is the question that matters
    once the full figure turns out not to fit.
    """
    penalty = CARDINALITY_PENALTY
    n_test = int(n_events * test_fraction)

    return [
        StageBudget(
            stage="download (auth.txt.gz + redteam.txt.gz)",
            # Deliberately *not* scaled by `n_events`. LANL serves one gzip
            # stream per file and no range of days within it, so a run over
            # two weeks still fetches all fifty-eight. Scaling this line by
            # the slice would understate the disk a partial run needs by
            # several gigabytes — exactly the error that shows up as `No space
            # left on device` after the download has already succeeded.
            disk_bytes=int(LANL_TOTAL_EVENTS * RAW_TEXT_BYTES_PER_EVENT * ASSUMED_GZIP_RATIO),
            peak_rss_bytes=64 * 1024**2,
            note="whole file: LANL serves no day ranges",
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
            peak_rss_bytes=6 * 1024**3,
            note="only the consumed diversity columns are computed",
        ),
        StageBudget(
            stage="train_eval (design matrix in RAM)",
            disk_bytes=64 * 1024**2,
            # Bounded by the larger of the fitting sample and one day of
            # scoring, not by the split. See `models.base.sample_for_fit` and
            # `train_eval.fit_and_score_all`.
            peak_rss_bytes=int(
                max(fit_sample_size, scoring_chunk_rows)
                * MODEL_MATRIX_BYTES_PER_EVENT
                * ESTIMATOR_COPY_FACTOR
            ),
            note="fit on a bounded sample, score in fixed-size chunks",
        ),
        StageBudget(
            stage="train_eval (bootstrap over the scored split)",
            disk_bytes=0,
            # The slim evaluation frame: identity columns plus one float64
            # score per model, plus the pre-sorted rank order the fast AUC-PR
            # path keeps for each of them.
            peak_rss_bytes=int(n_test * SCORED_FRAME_BYTES_PER_EVENT),
            note="slim score frame + one pre-sorted rank order per model",
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
        INTERIM_PARQUET_BYTES_PER_EVENT + FEATURE_STORE_BYTES_PER_EVENT
    ) * CARDINALITY_PENALTY
    # The whole compressed source lands on disk before any of it is converted,
    # however few days are kept, so it comes off the top rather than per event.
    download = LANL_TOTAL_EVENTS * RAW_TEXT_BYTES_PER_EVENT * ASSUMED_GZIP_RATIO
    usable = max(0, available_bytes - margin_bytes - download)
    return int(usable / per_event)


def max_events_for_memory(
    available_bytes: int,
    *,
    test_fraction: float = LANL_TEST_FRACTION,
) -> int:
    """Largest `n_events` the evaluation stage fits in `available_bytes`."""
    # The design matrix is bounded now, so what scales with the run is the
    # scored frame the bootstrap holds: the whole test split, slim.
    per_event = test_fraction * SCORED_FRAME_BYTES_PER_EVENT
    return int(available_bytes / per_event)


def available_memory_bytes() -> int:
    """Physical RAM installed, or 0 if psutil is unavailable."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return 0
    return int(psutil.virtual_memory().total)


def free_memory_bytes() -> int:
    """RAM actually available right now, or 0 if psutil is unavailable.

    Reported next to the installed total because the two answer different
    questions: the total says whether the run can ever work on this machine,
    the free figure says whether it can work *before closing the browser*.
    Printing only one of them turns "close some windows" into "buy a laptop".
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a declared dependency
        return 0
    return int(psutil.virtual_memory().available)


def available_disk_bytes(path: Path) -> int:
    return free_space_bytes(path)
