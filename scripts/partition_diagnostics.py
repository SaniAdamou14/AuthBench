"""Diagnostics of a labelled test partition that the run outputs do not carry.

Reads the labelled checkpoints `build_features` writes under `data/interim/` and
writes `diagnostics.json` next to the run's outputs: new (user, destination)
pairs per day within the window read, red-team events at the first second of
their pair, machine-account share, and campaigns truncated at the partition
boundary. Needs the checkpoints of the run being described.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def diagnostics(
    checkpoint: Path, first_day: int, last_day: int, campaigns: Path
) -> dict[str, object]:
    window = pl.scan_parquet(checkpoint).select(
        "time", "day", "src_user", "dst_computer", "is_malicious", "campaign_id"
    )
    first_seen = (
        window.group_by("src_user", "dst_computer")
        .agg(pl.col("time").min().alias("first_time"))
        .collect(engine="streaming")
    )
    new_pairs = (
        first_seen.group_by((pl.col("first_time") // 86_400).alias("day"))
        .agg(pl.len().alias("new_pairs"))
        .sort("day")
    )
    scored = window.filter(pl.col("day").is_between(first_day, last_day))
    malicious = (
        scored.filter(pl.col("is_malicious"))
        .collect()
        .join(first_seen, on=["src_user", "dst_computer"], how="left")
    )
    machine = scored.select(
        pl.len().alias("events"),
        pl.col("src_user").cast(pl.String).str.ends_with("$").sum().alias("machine_source_events"),
    ).collect()
    summary = pl.read_csv(campaigns).with_columns(
        (pl.col("start_time") // 86_400).alias("start_day"),
        (pl.col("end_time") // 86_400).alias("end_day"),
    )
    truncated = summary.filter(
        pl.col("start_day").is_between(first_day, last_day) & (pl.col("end_day") > last_day)
    )
    return {
        "window_days": [int(window.select(pl.col("day").min()).collect().item()), last_day],
        "new_pairs_per_day": new_pairs.to_dicts(),
        "malicious_at_first_second_of_pair_per_day": malicious.group_by("day")
        .agg(
            pl.len().alias("malicious"),
            (pl.col("time") == pl.col("first_time")).sum().alias("at_first_second"),
        )
        .sort("day")
        .to_dicts(),
        "events": int(machine["events"][0]),
        "machine_source_events": int(machine["machine_source_events"][0]),
        "malicious_from_machine_accounts": int(
            malicious["src_user"].cast(pl.String).str.ends_with("$").sum()
        ),
        "campaigns_truncated_at_partition_end": truncated.select(
            "campaign_id", "user_at_domain", "start_day", "end_day", "n_events"
        ).to_dicts(),
    }


def main() -> None:
    run = ROOT / "reports" / "lanl" / "wider_window_results"
    result = diagnostics(
        ROOT / "data" / "interim" / "_checkpoint_test.parquet", 11, 13, run / "campaign_summary.csv"
    )
    (run / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    json.dump(result, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
