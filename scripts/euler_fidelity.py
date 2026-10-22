"""Euler under its own protocol, to check the reimplementation against the published numbers (S6-09).

King and Huang (2022, table VI) report, for GCN + GRU in link prediction on LANL, AUC 0.9906
and AP 0.0155 over edges: every (30-minute snapshot, source computer, destination computer)
after training, an edge being anomalous when it carries a red-team authentication. Training
uses every snapshot before the first red-team event. This script reproduces that protocol on
every authentication event of the converted days (no AuthBench universe, split or feature
store) and writes the edge-level AUC and AP beside the published ones. It needs the whole
dataset and redteam.txt.gz: it is run on the rented machine, not in CI.

    python scripts/euler_fidelity.py data/interim/auth data/raw/redteam.txt.gz reports/lanl
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from authbench.models.euler import EulerScorer, snapshot_edges
from authbench.parse.clean import clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS

PUBLISHED = {"auc": 0.9906, "ap": 0.0155}
DAY = 86_400


def _peak_rss_gb() -> float | None:
    if sys.platform == "win32":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kilobytes on Linux


def log(message: str) -> None:
    peak = _peak_rss_gb()
    suffix = f" (peak RSS {peak:.1f} GB)" if peak is not None else ""
    print(f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {message}{suffix}", flush=True)


def edge_level(auth: pl.LazyFrame, redteam: pl.LazyFrame, snapshot: int) -> pl.DataFrame:
    """One row per (snapshot, source, destination), labelled by the red-team events it holds.

    The distinct edges come from `snapshot_edges`, a day at a time: a single `unique` over the
    billion LANL events was OOM-killed at 50.8 GB.
    """
    attacks = (
        redteam.select(
            (pl.col("time") // snapshot).cast(pl.Int64).alias("snapshot"),
            pl.col("src_computer").cast(pl.Utf8).alias("src"),
            pl.col("dst_computer").cast(pl.Utf8).alias("dst"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("is_malicious"))
        .collect()
    )
    return (
        snapshot_edges(auth, snapshot)
        .drop("len")
        .join(attacks, on=["snapshot", "src", "dst"], how="left")
        .with_columns(pl.col("is_malicious").fill_null(False))
        .rename({"src": "src_computer", "dst": "dst_computer"})
    )


def score_edges(model: EulerScorer, edges: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Scores and labels of the edges, a day at a time (scoring is row-local, so this is exact).

    All test edges at once gather two float64 arrays of 108 million rows by 16 embeddings.
    """
    snapshot = model.snapshot_seconds
    day = pl.col("snapshot") * snapshot // DAY
    first, last = edges.select(day.min().alias("first"), day.max().alias("last")).row(0)
    scores, labels = [], []
    for d in range(first, last + 1):
        chunk = edges.filter(day == d)
        if chunk.height:
            events = chunk.select(
                (pl.col("snapshot") * snapshot).alias("time"), "src_computer", "dst_computer"
            )
            scores.append(model.score(events.lazy()).to_numpy())
            labels.append(chunk["is_malicious"].to_numpy())
    return np.concatenate(scores), np.concatenate(labels)


def run(auth_dir: Path, redteam_path: Path, out_dir: Path, model: EulerScorer) -> dict[str, object]:
    auth = pl.scan_parquet(auth_dir / "**" / "*.parquet").select(
        "time", "src_computer", "dst_computer"
    )
    redteam, _ = clean_redteam(
        pl.scan_csv(redteam_path, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    )
    first_attack = int(redteam.select(pl.col("time").min()).collect().item())
    log(f"first red-team event at t={first_attack}; fitting")
    model.fit(auth.filter(pl.col("time") < first_attack))
    log(f"fitted in {len(model.history)} epochs; preparing the timeline")
    model.prepare([auth])
    log("prepared; building the test edges")
    edges = edge_level(auth, redteam, model.snapshot_seconds).filter(
        pl.col("snapshot") * model.snapshot_seconds >= first_attack
    )
    log(f"{edges.height} test edges, {int(edges['is_malicious'].sum())} anomalous; scoring")
    scores, labels = score_edges(model, edges)
    n_edges = edges.height
    del edges
    log("scored; computing AUC and AP")
    result = {
        "protocol": "King and Huang (2022): link prediction, edges per 30-minute snapshot, "
        "training before the first red-team event",
        "first_red_team_event": first_attack,
        "test_edges": n_edges,
        "anomalous_edges": int(labels.sum()),
        "auc": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < labels.size else None,
        "ap": float(average_precision_score(labels, scores)) if labels.any() else None,
        "published": PUBLISHED,
        "training_epochs": len(model.history),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "euler_fidelity.json").write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {out_dir / 'euler_fidelity.json'}")
    return result


def main() -> None:
    auth_dir, redteam_path, out_dir = (Path(a) for a in sys.argv[1:4])
    print(json.dumps(run(auth_dir, redteam_path, out_dir, EulerScorer()), indent=2))


if __name__ == "__main__":
    main()
