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
from pathlib import Path

import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from authbench.models.euler import COLUMNS, EulerScorer
from authbench.parse.clean import clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS

PUBLISHED = {"auc": 0.9906, "ap": 0.0155}


def edge_level(auth: pl.LazyFrame, redteam: pl.LazyFrame, snapshot: int) -> pl.LazyFrame:
    """One row per (snapshot, source, destination), labelled by the red-team events it holds."""
    keys = [
        (pl.col("time") // snapshot).alias("snapshot"),
        pl.col("src_computer").cast(pl.Utf8),
        pl.col("dst_computer").cast(pl.Utf8),
    ]
    attacks = redteam.select(*keys).unique().with_columns(pl.lit(True).alias("is_malicious"))
    return (
        auth.select(*keys)
        .unique()
        .join(attacks, on=["snapshot", "src_computer", "dst_computer"], how="left")
        .with_columns(pl.col("is_malicious").fill_null(False))
    )


def run(auth_dir: Path, redteam_path: Path, out_dir: Path, model: EulerScorer) -> dict[str, object]:
    auth = pl.scan_parquet(auth_dir / "**" / "*.parquet").select(COLUMNS)
    redteam, _ = clean_redteam(
        pl.scan_csv(redteam_path, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    )
    first_attack = int(redteam.select(pl.col("time").min()).collect().item())
    model.fit(auth.filter(pl.col("time") < first_attack))
    model.prepare([auth])
    edges = (
        edge_level(auth, redteam, model.snapshot_seconds)
        .filter(pl.col("snapshot") * model.snapshot_seconds >= first_attack)
        .collect()
    )
    scored = edges.with_columns((pl.col("snapshot") * model.snapshot_seconds).alias("time"))
    scores = model.score(scored.lazy()).to_numpy()
    labels = edges["is_malicious"].to_numpy()
    result = {
        "protocol": "King and Huang (2022): link prediction, edges per 30-minute snapshot, "
        "training before the first red-team event",
        "first_red_team_event": first_attack,
        "test_edges": int(edges.height),
        "anomalous_edges": int(labels.sum()),
        "auc": float(roc_auc_score(labels, scores)) if 0 < labels.sum() < labels.size else None,
        "ap": float(average_precision_score(labels, scores)) if labels.any() else None,
        "published": PUBLISHED,
        "training_epochs": len(model.history),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "euler_fidelity.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    auth_dir, redteam_path, out_dir = (Path(a) for a in sys.argv[1:4])
    print(json.dumps(run(auth_dir, redteam_path, out_dir, EulerScorer()), indent=2))


if __name__ == "__main__":
    main()
