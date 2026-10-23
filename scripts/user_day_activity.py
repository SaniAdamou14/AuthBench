"""User-day activity on the test days with red-team activity, for label-free unit baselines.

The harness ranks user-days (the source user, without its domain, as `evaluate.statistics.UNITS`
keys them) by the highest score of their events. A ranking that reads no detector, by how many
computers a user reached that day, is the baseline such a unit needs, and the number of events of
each user-day is what a random-ranking null of max-aggregated scores depends on. Neither is in
the run outputs, so this script recomputes them from the authentication log.

For each protocol it writes, under `--out`:

- `user_days_<protocol>.csv.gz`: one row per (day, src_user) of the test universe, with its
  events, distinct destination and source computers, failures, and first event (time, event_id);
- `positives_<protocol>.csv`: every labelled positive of the test universe with its identity
  (event_id, day, time, user, domain, source and destination computers);
- `user_day_activity.json`: provenance and the per-day counts that `scripts/paper_assets.py`
  checks against the run's `day_sizes.csv` and positives ledger.

Rows are parsed and typed by the pipeline's own `_parse_block` and `clean_auth`, with the same
`event_id` (the row's index in auth.txt) and the same universe filter as the run. Days on disk are
read from `--interim` (the day partitions of `to_parquet`); later days are streamed from `--stream`
(a URL of auth.txt.gz), decompressed in memory and never written, up to the last day needed. The
last day on disk before them is streamed too and must match its partition. Only days with a
red-team event matter for campaign recall: harmonised days 1-29, temporal days 11-13.

    python scripts/user_day_activity.py --interim data/interim/auth \\
        --redteam redteam.txt.gz --out reports/runs/<run>/diagnostics/user_days \\
        --stream https://lanl.ma.ic.ac.uk/data/cyber1/auth.txt.gz
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import time as clock
import zlib
from collections import defaultdict, deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import polars as pl
import requests

from authbench.ingest.to_parquet import _parse_block
from authbench.parse.clean import clean_auth, clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS
from authbench.pipeline.build_features import universe_filter

logger = logging.getLogger("user_day_activity")

HOUR = 3600
DAY = 86400
TEXT = ["src_user", "src_domain", "src_computer", "dst_computer"]
# name: (universe, first second of the test, the test days holding red-team events)
PROTOCOLS = {
    "harmonised": ("remote_logon", 41 * HOUR, range(1, 30)),
    "temporal": ("all", 11 * DAY, range(11, 14)),
}
REDTEAM_KEY = ["time", *TEXT]
BLOCK_BYTES = 64 * 1024 * 1024
# Rows of a day are compared with the partition on disk through this fingerprint.
FINGERPRINT_MODULUS = 1_000_003


def read_redteam(path: Path) -> pl.DataFrame:
    data = path.read_bytes()
    raw = pl.read_csv(
        gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data,
        has_header=False,
        new_columns=RAW_REDTEAM_COLUMNS,
        schema_overrides=dict.fromkeys(RAW_REDTEAM_COLUMNS, pl.Utf8),
    ).with_columns(pl.col("time").cast(pl.Int64))
    typed, _ = clean_redteam(raw.lazy())
    rows: pl.DataFrame = typed.select(
        pl.col("time").cast(pl.Int64),
        pl.col("user").cast(pl.Utf8).alias("src_user"),
        pl.col("domain").cast(pl.Utf8).alias("src_domain"),
        pl.col("src_computer").cast(pl.Utf8),
        pl.col("dst_computer").cast(pl.Utf8),
    ).collect()
    return rows


def plain(frame: pl.LazyFrame) -> pl.LazyFrame:
    """The columns used here, identifiers as text so frames of different blocks agree."""
    return frame.select(
        pl.col("event_id").cast(pl.Int64),
        pl.col("time").cast(pl.Int64),
        pl.col("day").cast(pl.Int64),
        *(pl.col(c).cast(pl.Utf8) for c in TEXT),
        pl.col("auth_orientation").cast(pl.Utf8),
        pl.col("success"),
    )


def in_test(frame: pl.LazyFrame, protocol: str) -> pl.LazyFrame:
    universe, start, days = PROTOCOLS[protocol]
    return frame.filter(
        universe_filter(universe)
        & (pl.col("time") >= start)
        & pl.col("day").is_between(days.start, days.stop - 1)
    )


def fingerprint(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Row count, event_id range and an order-free hash sum of every row."""
    return frame.select(
        pl.len().alias("rows"),
        pl.col("event_id").min().alias("first_event_id"),
        pl.col("event_id").max().alias("last_event_id"),
        (pl.struct(pl.all()).hash(seed=0) % FINGERPRINT_MODULUS).cast(pl.Int64).sum().alias("hash"),
    )


def counts(frame: pl.LazyFrame) -> pl.LazyFrame:
    """The additive part of a user-day: events, failures, first event."""
    return frame.group_by("day", "src_user").agg(
        pl.len().alias("events"),
        (~pl.col("success")).sum().cast(pl.Int64).alias("failures"),
        pl.col("time").min().alias("first_time"),
        pl.col("event_id").min().alias("first_event_id"),
    )


def pairs(frame: pl.LazyFrame, column: str) -> pl.LazyFrame:
    return frame.select("day", "src_user", column).unique()


def merge(
    partial: list[pl.DataFrame], dst: list[pl.DataFrame], src: list[pl.DataFrame]
) -> pl.DataFrame:
    """User-days of one day from per-block parts: sums, minima and distinct counts."""
    merged = (
        pl.concat(partial)
        .group_by("day", "src_user")
        .agg(
            pl.col("events").sum(),
            pl.col("failures").sum(),
            pl.col("first_time").min(),
            pl.col("first_event_id").min(),
        )
    )
    for name, frames in (("distinct_dst", dst), ("distinct_src", src)):
        distinct = pl.concat(frames).unique().group_by("day", "src_user").agg(pl.len().alias(name))
        merged = merged.join(distinct, on=["day", "src_user"], how="left")
    return merged.select(
        "day",
        "src_user",
        "events",
        pl.col("distinct_dst").cast(pl.Int64),
        pl.col("distinct_src").cast(pl.Int64),
        "failures",
        "first_time",
        "first_event_id",
    ).sort("day", "src_user")


def positives(frame: pl.LazyFrame, redteam: pl.DataFrame) -> pl.LazyFrame:
    return frame.join(redteam.lazy(), on=REDTEAM_KEY, how="semi").select(
        "event_id", "day", *REDTEAM_KEY
    )


def interim_day(root: Path, day: int) -> pl.LazyFrame:
    return plain(pl.scan_parquet(root / f"day={day}" / "*.parquet"))


def fetch_range(url: str, start: int, end: int, retries: int = 50) -> bytes:
    """Bytes [start, end] of `url`, retried until they arrive whole."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise requests.HTTPError(
                    f"HTTP {response.status_code}, {len(response.content)} bytes at {start}"
                )
            return response.content
        except (requests.RequestException, ConnectionError) as exc:
            if attempt == retries:
                raise
            logger.warning("range %d-%d failed (%s); retrying", start, end, exc)
            clock.sleep(min(60, 5 * attempt))
    raise AssertionError("unreachable")


def decompressed_stream(url: str, chunk: int = 8 << 20, workers: int = 6) -> Iterator[bytes]:
    """auth.txt.gz decompressed on the fly, its byte ranges fetched in parallel, decoded in order.

    One connection to the mirror carries about 0.2 MB/s; several in parallel carry more. At most
    `2 * workers` chunks are held ahead of the decoder.
    """
    size = int(requests.head(url, timeout=120).headers["Content-Length"])
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    started, done, reported = clock.monotonic(), 0, 0
    pool = ThreadPoolExecutor(workers)
    pending: deque[Future[bytes]] = deque()
    offset = 0
    try:
        while offset < size or pending:
            while offset < size and len(pending) < 2 * workers:
                pending.append(pool.submit(fetch_range, url, offset, min(offset + chunk, size) - 1))
                offset += chunk
            data = pending.popleft().result()
            done += len(data)
            out = inflate.decompress(data)
            if out:
                yield out
            if done - reported >= 256 << 20:
                reported = done
                rate = done / max(1.0, clock.monotonic() - started) / 1e6
                logger.info("%.2f of %.2f GB read (%.2f MB/s)", done / 1e9, size / 1e9, rate)
        tail = inflate.flush()
        if tail:
            yield tail
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def line_blocks(chunks: Iterator[bytes], block_bytes: int = BLOCK_BYTES) -> Iterator[bytes]:
    buffer = bytearray()
    for chunk in chunks:
        buffer += chunk
        if len(buffer) >= block_bytes:
            cut = buffer.rfind(b"\n")
            yield bytes(buffer[: cut + 1])
            del buffer[: cut + 1]
    if buffer.strip():
        yield bytes(buffer)


def stream_blocks(url: str, first: int, last: int) -> Iterator[pl.DataFrame]:
    """Typed rows of days [first, last], block by block, parsed exactly as `to_parquet` does.

    The source is in time order: a block whose last row precedes `first` is only counted (its
    rows still advance `event_id`), and reading stops at the first block past `last`.
    """
    id_offset = 0
    for block in line_blocks(decompressed_stream(url)):
        last_line = block.rstrip(b"\n").rsplit(b"\n", 1)[-1]
        if int(last_line.split(b",", 1)[0]) < first * DAY:
            id_offset += block.count(b"\n")
            continue
        raw = _parse_block(block)
        typed, _ = clean_auth(raw.lazy(), id_offset=id_offset)
        id_offset += raw.height
        rows = plain(typed).collect()
        if int(rows["day"].min()) > last:  # type: ignore[arg-type]
            return
        yield rows.filter(pl.col("day").is_between(first, last))


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--interim", type=Path, required=True)
    parser.add_argument("--redteam", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stream", help="URL of auth.txt.gz, for the days not in --interim")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    redteam = read_redteam(args.redteam)
    on_disk = {int(p.name.split("=")[1]) for p in args.interim.glob("day=*")}
    units: dict[str, list[pl.DataFrame]] = defaultdict(list)
    labelled: dict[str, list[pl.DataFrame]] = defaultdict(list)
    sources: dict[str, dict[int, str]] = defaultdict(dict)
    missing: set[int] = set()

    for protocol, (_, _, days) in PROTOCOLS.items():
        for day in days:
            if day not in on_disk:
                missing.add(day)
                continue
            test = in_test(interim_day(args.interim, day), protocol)
            units[protocol].append(
                merge(
                    [counts(test).collect(engine="streaming")],
                    [pairs(test, "dst_computer").collect(engine="streaming")],
                    [pairs(test, "src_computer").collect(engine="streaming")],
                )
            )
            labelled[protocol].append(positives(test, redteam).collect(engine="streaming"))
            sources[protocol][day] = "interim"
            logger.info("%s day %d from disk", protocol, day)

    checked: dict[str, object] = {}
    if missing:
        if not args.stream:
            raise SystemExit(f"Days {sorted(missing)} are not in {args.interim}; pass --stream.")
        first, last = min(missing), max(missing)
        check_day = max((d for d in on_disk if d < first), default=None)
        streamed_protocols = [
            p for p, (_, _, days) in PROTOCOLS.items() if any(d in missing for d in days)
        ]
        parts: dict[tuple[str, int], tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
        check_parts: list[pl.DataFrame] = []
        for rows in stream_blocks(args.stream, check_day if check_day is not None else first, last):
            if check_day is not None:
                check_parts.append(
                    fingerprint(rows.filter(pl.col("day") == check_day).lazy()).collect()
                )
            for protocol in streamed_protocols:
                test = in_test(rows.lazy(), protocol).filter(pl.col("day").is_in(sorted(missing)))
                for (day,), part in test.collect().partition_by("day", as_dict=True).items():
                    lazy = part.lazy()
                    bucket = parts[(protocol, int(day))]
                    bucket[0].append(counts(lazy).collect())
                    bucket[1].append(pairs(lazy, "dst_computer").collect())
                    bucket[2].append(pairs(lazy, "src_computer").collect())
                    labelled[protocol].append(positives(lazy, redteam).collect())
        if check_day is not None:
            streamed = pl.concat(check_parts).select(
                pl.col("rows").sum(),
                pl.col("first_event_id").min(),
                pl.col("last_event_id").max(),
                pl.col("hash").sum(),
            )
            disk = fingerprint(interim_day(args.interim, check_day)).collect(engine="streaming")
            if streamed.row(0) != disk.row(0):
                raise SystemExit(
                    f"Streamed day {check_day} {streamed.row(0)} != disk {disk.row(0)}."
                )
            checked = {
                "day": check_day,
                "fingerprint": dict(zip(disk.columns, disk.row(0), strict=True)),
            }
            logger.info("day %d streamed and on disk agree: %s", check_day, disk.row(0))
        for (protocol, day), (partial, dst, src) in sorted(parts.items()):
            units[protocol].append(merge(partial, dst, src))
            sources[protocol][day] = "stream"

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "redteam_sha256": hashlib.sha256(args.redteam.read_bytes()).hexdigest(),
        "redteam_distinct_rows": redteam.height,
        "stream": args.stream,
        "stream_check": checked,
    }
    for protocol in PROTOCOLS:
        table = pl.concat(units[protocol]).sort("day", "src_user")
        positive = pl.concat(labelled[protocol]).sort("day", "time", "event_id")
        # No name or time in the gzip header, so the file is the same on every run.
        with (
            (args.out / f"user_days_{protocol}.csv.gz").open("wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle,
        ):
            handle.write(table.write_csv().encode())
        positive.write_csv(args.out / f"positives_{protocol}.csv")
        per_day = (
            table.group_by("day")
            .agg(pl.len().alias("user_days"), pl.col("events").sum().alias("events"))
            .join(positive.group_by("day").agg(pl.len().alias("positives")), on="day", how="left")
            .with_columns(pl.col("positives").fill_null(0))
            .sort("day")
        )
        summary[protocol] = {
            "day_sources": {str(d): s for d, s in sorted(sources[protocol].items())},
            "positives": positive.height,
            "redteam_rows_matched": positive.select(REDTEAM_KEY).unique().height,
            "days": per_day.to_dicts(),
        }
    (args.out / "user_day_activity.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
