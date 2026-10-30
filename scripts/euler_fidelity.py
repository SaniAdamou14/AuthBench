"""Euler under its own protocol, to check the reimplementation against the published numbers.

King and Huang (2022, table VI) report, for GCN + GRU in link prediction on LANL, AUC 0.9906
and AP 0.0155 over edges: every (30-minute snapshot, source computer, destination computer)
after training, an edge being anomalous when it carries a red-team authentication. Training
uses every snapshot before the first red-team event. This script reproduces that protocol on
every authentication event of the converted days (no AuthBench universe, split or feature
store) and writes the edge-level AUC and AP beside the published ones. It needs the whole
dataset and redteam.txt.gz: it is run on the rented machine, not in CI.

Configurations, each written to `euler_fidelity_<config>.json` with one result per seed and
their mean and standard deviation (the paper averages 5 independent runs):

- `ntlm`, the authors' loader: their code (lanl_experiments/loaders/split.py) keeps only the
  lines of auth.txt that contain "NTLM" ("Some filtering for better FPR/less Kerb noise") and
  load_lanl.py skips self-loops. The paper does not say so (section VI-A: "for all
  authentications"), but its table V counts 45,871,390 events and 518 anomalous edges, far
  from the 1.05 billion events of auth.txt. Measured on LANL: every red-team edge is NTLM, and
  the filter leaves 14.4 million of the 99.4 million test edges.
- `ntlm_vocabulary`, and every node of the timeline its own identifier, as their code does
  (load_lanl.py: one-hot features over the whole node map), negatives sampled among them all.
- `ntlm_code`, and their code's defaults for the learning rate and the patience (run.py:
  0.005 and 5; the paper's text says ten epochs of patience).
- `all`, every authentication event, self-loops included: the graph AuthBench's M5c sees.

Not reproduced: their test snapshots start at the first red-team event (t=150885) rather than
on multiples of 30 minutes.

    python scripts/euler_fidelity.py data/interim/auth data/raw/redteam.txt.gz reports/lanl \
        [ntlm ntlm_vocabulary ntlm_code all] [--seeds 42,43,44,45,46]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from authbench.models.euler import EulerScorer, snapshot_edges
from authbench.parse.clean import clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS

PUBLISHED = {"auc": 0.9906, "ap": 0.0155}
DAY = 86_400
# The fields of a raw auth.txt line, which the authors' filter searches for "NTLM".
RAW_FIELDS = [
    "src_user",
    "src_domain",
    "dst_user",
    "dst_domain",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success",
]
EVENTS = {
    "ntlm": "NTLM authentications only, self-loops dropped (the authors' loader)",
    "all": "every authentication event, self-loops included (AuthBench's graph)",
}
CONFIGS: dict[str, dict[str, Any]] = {
    "ntlm": {"events": "ntlm", "vocabulary": False, "params": {}},
    "ntlm_vocabulary": {"events": "ntlm", "vocabulary": True, "params": {}},
    "ntlm_code": {
        "events": "ntlm",
        "vocabulary": True,
        "params": {"learning_rate": 0.005, "patience": 5},
    },
    "all": {"events": "all", "vocabulary": False, "params": {}},
}


def _peak_rss_gb() -> float | None:
    if sys.platform == "win32":
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kilobytes on Linux


def log(message: str) -> None:
    peak = _peak_rss_gb()
    suffix = f" (peak RSS {peak:.1f} GB)" if peak is not None else ""
    print(f"[{datetime.now(UTC):%Y-%m-%dT%H:%M:%SZ}] {message}{suffix}", flush=True)


def select_events(auth: pl.LazyFrame, variant: str) -> pl.LazyFrame:
    """The events a variant keeps, as (time, source computer, destination computer)."""
    if variant == "ntlm":
        fields = [c for c in RAW_FIELDS if c in auth.collect_schema().names()]
        line = pl.concat_str(
            [pl.col(c).cast(pl.Utf8).fill_null("") for c in fields], separator=","
        ).str.to_uppercase()
        auth = auth.filter(
            line.str.contains("NTLM", literal=True)
            & (pl.col("src_computer").cast(pl.Utf8) != pl.col("dst_computer").cast(pl.Utf8))
        )
    elif variant != "all":
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(EVENTS)}.")
    return auth.select("time", "src_computer", "dst_computer")


def events(auth_dir: Path, variant: str, scratch: Path) -> pl.LazyFrame:
    """A variant's events. `ntlm` is filtered a day at a time into `scratch` first: as one lazy
    query over the whole dataset, its search of ten text columns was OOM-killed at a 24 GB cap."""
    if variant == "all":
        return select_events(pl.scan_parquet(auth_dir / "**" / "*.parquet"), variant)
    for day in sorted(auth_dir.glob("day=*")):
        (scratch / day.name).mkdir(parents=True)
        (
            select_events(pl.scan_parquet(day / "*.parquet"), variant)
            .with_columns(pl.col("src_computer", "dst_computer").cast(pl.Utf8))
            .collect()
            .write_parquet(scratch / day.name / "part.parquet")
        )
    return pl.scan_parquet(scratch / "*" / "*.parquet")


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


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    kept = [v for v in values if v is not None]
    if not kept:
        return None, None
    return float(np.mean(kept)), float(np.std(kept, ddof=1)) if len(kept) > 1 else None


def run(
    auth_dir: Path,
    redteam_path: Path,
    out_dir: Path,
    config: str = "all",
    seeds: tuple[int, ...] = (42,),
    **overrides: Any,
) -> dict[str, object]:
    """One configuration, fitted and scored once per seed; `overrides` replace model settings."""
    spec = CONFIGS[config]
    params = {**spec["params"], **overrides}
    snapshot = EulerScorer(**params).snapshot_seconds
    redteam, _ = clean_redteam(
        pl.scan_csv(redteam_path, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    )
    first_attack = int(redteam.select(pl.col("time").min()).collect().item())
    runs = []
    with tempfile.TemporaryDirectory(prefix="euler_fidelity_") as scratch:
        auth = events(auth_dir, spec["events"], Path(scratch))
        n_events = int(auth.select(pl.len()).collect().item())
        vocabulary = None
        if spec["vocabulary"]:
            nodes: set[str] = set()
            for column in ("src_computer", "dst_computer"):
                nodes |= set(auth.select(pl.col(column).cast(pl.Utf8).unique()).collect()[column])
            vocabulary = sorted(nodes)
        log(
            f"config {config}: {n_events} events, vocabulary "
            f"{len(vocabulary) if vocabulary else 'from training'}; first red-team event at "
            f"t={first_attack}"
        )
        # Every event from the first red-team one on, as the authors' loader reads the test
        # range: the snapshot holding it is kept, without its earlier events.
        edges = edge_level(auth.filter(pl.col("time") >= first_attack), redteam, snapshot)
        n_anomalous = int(edges["is_malicious"].sum())
        log(f"{edges.height} test edges, {n_anomalous} anomalous")
        for seed in seeds:
            cache = Path(scratch) / f"embeddings_{seed}"
            model = EulerScorer(**params, seed=seed, vocabulary=vocabulary, cache_dir=str(cache))
            model.fit(auth.filter(pl.col("time") < first_attack))
            model.prepare([auth])
            scores, labels = score_edges(model, edges)
            single = 0 < labels.sum() < labels.size
            runs.append(
                {
                    "seed": seed,
                    "auc": float(roc_auc_score(labels, scores)) if single else None,
                    "ap": float(average_precision_score(labels, scores)) if labels.any() else None,
                    "training_epochs": len(model.history),
                }
            )
            del model, scores
            shutil.rmtree(cache)
            log(f"seed {seed}: {runs[-1]}")
    auc_mean, auc_std = _mean_std([r["auc"] for r in runs])
    ap_mean, ap_std = _mean_std([r["ap"] for r in runs])
    result: dict[str, object] = {
        "protocol": "King and Huang (2022): link prediction, edges per 30-minute snapshot, "
        "training before the first red-team event",
        "config": config,
        "events": EVENTS[spec["events"]],
        "vocabulary": "every node of the timeline"
        if spec["vocabulary"]
        else "training nodes, then one shared unseen node",
        "params": params,
        "events_kept": n_events,
        "first_red_team_event": first_attack,
        "test_edges": edges.height,
        "anomalous_edges": n_anomalous,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "ap_mean": ap_mean,
        "ap_std": ap_std,
        "runs": runs,
        "published": PUBLISHED,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"euler_fidelity_{config}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    log(f"wrote {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("auth_dir", type=Path)
    parser.add_argument("redteam", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("configs", nargs="*", help=f"any of {list(CONFIGS)} (default: all)")
    parser.add_argument("--seeds", default="42", help="comma-separated, e.g. 42,43,44,45,46")
    args = parser.parse_args()
    unknown = sorted(set(args.configs) - set(CONFIGS))
    if unknown:
        parser.error(f"unknown configurations {unknown}; expected some of {list(CONFIGS)}")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    for config in args.configs or list(CONFIGS):
        result = run(args.auth_dir, args.redteam, args.out_dir, config, seeds)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
