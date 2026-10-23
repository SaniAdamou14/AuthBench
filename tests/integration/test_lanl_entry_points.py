"""The entry points that produced the paper's numbers, run end to end on the demo sample.

`authbench demo` re-implements the pipeline for speed; the LANL run used `build_features.main`
and `train_eval.main` instead, through Hydra. This test composes the real configuration with
both protocols, converts the demo log as `to_parquet` converts LANL's, runs both stages, and
checks that their outputs agree with each other: every labelled test event is in the positives
ledger of every model, the day sizes cover the test partition, and each model's reported first
detection is the smallest rank in its ledger.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize_config_dir

from authbench.ingest.to_parquet import to_parquet_partitioned
from authbench.pipeline import build_features, train_eval
from authbench.pipeline.build_features import CONF_DIR

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("lanl_entry_points")
    (base / "raw").mkdir()
    (base / "raw" / "redteam.txt.gz").write_bytes(
        gzip.compress((ROOT / "data" / "demo" / "redteam_demo.txt").read_bytes())
    )
    to_parquet_partitioned(
        ROOT / "data" / "demo" / "auth_demo.txt",
        base / "interim" / "auth",
        check_disk_space=False,
    )
    return base


def _run(base: Path, split: str) -> Path:
    out = base / split
    overrides = [
        "dataset=demo",
        f"split={split}",
        f"paths.raw_dir={base / 'raw'}",
        f"paths.interim_dir={base / 'interim'}",
        f"paths.processed_dir={out / 'processed'}",
        f"paths.reports_dir={out / 'reports'}",
        f"paths.figures_dir={out / 'reports' / 'figures'}",
        f"paths.tables_dir={out / 'reports' / 'tables'}",
        f"mlflow.tracking_uri=sqlite:///{(out / 'mlruns.db').as_posix()}",
        "eval.bootstrap.n_resamples=200",
        "eval.ledger.n_resamples=100",
        "eval.ledger.n_simulations=100",
        "eval.tuning.n_configs=3",
        "eval.fit_variance.seeds=[42,7]",
    ]
    if split == "harmonised":
        # Its test runs to LANL's last hour; the demo log holds days 0 to 11.
        overrides.append("split.test_hours=[41,288]")
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    build_features.main(cfg)
    train_eval.main(cfg)
    return out


@pytest.mark.parametrize("split", ["demo_temporal", "harmonised"])
def test_build_features_then_train_eval_agree(converted: Path, split: str) -> None:
    out = _run(converted, split)
    tables = out / "reports" / "tables"
    for name in (
        "positives_ledger.csv",
        "day_sizes.csv",
        "budget_summary.json",
        "fit_variance.json",
    ):
        assert (tables / name).exists(), name

    (store,) = (out / "processed" / "features").iterdir()
    test = pl.read_parquet(store / "test" / "*.parquet")
    ledger = pl.read_csv(tables / "positives_ledger.csv")
    sizes = pl.read_csv(tables / "day_sizes.csv")
    assert int(test["is_malicious"].sum()) > 0
    assert sizes["events"].sum() == test.height
    assert sizes["positives"].sum() == int(test["is_malicious"].sum())

    labelled = set(test.filter(pl.col("is_malicious"))["event_id"].to_list())
    summary = json.loads((out / "reports" / "metrics_summary.json").read_text())
    assert summary, "no model was evaluated"
    for row in summary:
        rows = ledger.filter(pl.col("model") == row["model"])
        assert set(rows["event_id"].to_list()) == labelled, row["model"]
        assert row["operational"]["budget_for_first_detection"] == rows["rank"].min()

    has_validation = (store / "val").exists()
    assert has_validation == (split == "demo_temporal")
    if split == "demo_temporal":
        assert (tables / "r2_semi_supervised" / "positives_ledger.csv").exists()
