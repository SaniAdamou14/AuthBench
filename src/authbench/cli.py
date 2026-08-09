"""AuthBench command-line interface.

Every subcommand is a thin wrapper around `authbench.*` library code — the
CLI itself holds no business logic, so everything here is equally usable
from a notebook or a DVC stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from authbench.evaluate.budget import DEFAULT_BUDGETS, compute_budget_curve
from authbench.evaluate.metrics import auc_pr
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget
from authbench.features.event import compute_f1, fit_frequency_encoding
from authbench.features.history import compute_f2
from authbench.features.novelty import compute_f3
from authbench.features.temporal import calibrate_night_window, compute_f4
from authbench.ingest.download import download_with_resume, fetch_lanl_fence_token, lanl_file_url
from authbench.ingest.to_parquet import to_parquet_partitioned, verify_row_count
from authbench.label.redteam_join import (
    attach_campaign_id,
    campaign_summary,
    group_into_campaigns,
    label_auth_events,
    labeling_report,
)
from authbench.models.classical import ECODScorer, IsolationForestScorer
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.rules import RulesScorer
from authbench.models.stats import PairRarityScorer, PCAReconstructionScorer
from authbench.parse.clean import clean_auth, clean_redteam, data_quality_report
from authbench.split.temporal import (
    TemporalSplitConfig,
    get_test_split,
    get_train_split,
    get_val_split,
    verify_temporal_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("authbench.cli")

app = typer.Typer(
    help="AuthBench — leak-free, alert-budget-constrained auth-log anomaly benchmark."
)
data_app = typer.Typer(help="Data acquisition and conversion.")
app.add_typer(data_app, name="data")

console = Console()

F1_TO_F4_MODEL_FEATURES = [
    "is_success",
    "auth_type_is_null",
    "logon_type_is_null",
    "src_user_is_machine",
    "src_dst_user_same",
    "src_dst_computer_same",
    "domain_crossing",
    "auth_type_freq",
    "logon_type_freq",
    "auth_orientation_freq",
    "src_user_1h_n_events",
    "src_user_1h_failure_ratio",
    "src_user_1d_n_events",
    "src_user_1d_failure_ratio",
    "pair_is_new",
    "pair_global_rarity",
    "user_new_host_count_24h",
    "host_new_user_count_24h",
    "hour_sin",
    "hour_cos",
    "hour_deviation_from_profile",
]


@data_app.command("download")
def data_download(
    dataset: str = typer.Option("lanl", help="Dataset name (currently: lanl)."),
    out_dir: Path = typer.Option(Path("data/raw"), help="Destination directory."),
    email: str | None = typer.Option(
        None,
        envvar="AUTHBENCH_LANL_EMAIL",
        help="Email for LANL's data-use form (csr.lanl.gov). Also read from AUTHBENCH_LANL_EMAIL.",
    ),
    usage: str = typer.Option(
        "Academic research: anomaly detection benchmark on authentication logs (AuthBench project).",
        envvar="AUTHBENCH_LANL_USAGE",
        help="Usage statement for LANL's data-use form.",
    ),
) -> None:
    """US-101: resumable, checksum-verified download of the raw LANL files.

    LANL gates the raw files behind a click-through data-use form rather than
    a stable static URL (`ingest.download.fetch_lanl_fence_token`); `email`
    and `usage` are submitted to that form, never stored in `conf/`.

    `email` is an env-var-backed option rather than a required flag so that
    `dvc repro` and CI can run the stage unattended without an address being
    committed into `dvc.yaml`.
    """
    from omegaconf import OmegaConf

    if not email:
        raise typer.BadParameter(
            "LANL's data-use form requires an email address. Pass --email, or set "
            "AUTHBENCH_LANL_EMAIL in your environment (what `dvc repro` expects).",
            param_hint="--email",
        )

    cfg = OmegaConf.load(f"conf/dataset/{dataset}.yaml")
    token = fetch_lanl_fence_token(email, usage)

    for key, filename in cfg.fence.filenames.items():
        url = lanl_file_url(token, filename)
        dest = out_dir / filename
        expected = cfg.sha256.get(key)
        result = download_with_resume(url, dest, expected)
        console.print(
            f"[green]{key}[/]: {dest} (skipped={result.skipped}, resumed={result.resumed}, "
            f"sha256={result.sha256})"
        )
        if expected is None:
            console.print(
                f"  [yellow]No checksum recorded yet for '{key}' — add sha256.{key}: "
                f"{result.sha256} to conf/dataset/{dataset}.yaml to verify future runs.[/]"
            )


@data_app.command("to-parquet")
def data_to_parquet(
    src: Path = typer.Argument(..., help="Path to auth.txt(.gz)."),
    out_dir: Path = typer.Option(Path("data/interim/auth"), help="Output Parquet directory."),
    expected_rows: int | None = typer.Option(None, help="Abort if converted row count mismatches."),
) -> None:
    """US-102: stream raw text into day-partitioned, ZSTD-compressed Parquet."""
    report = to_parquet_partitioned(src, out_dir)
    console.print(
        f"Converted {report.n_rows_out:,}/{report.n_rows_in:,} rows in {report.n_batches} "
        f"batches, peak RSS {report.peak_rss_bytes / 1e9:.2f} GB."
    )
    if expected_rows is not None:
        verify_row_count(report, expected_rows)
        console.print("[green]Row count verified.[/]")


@data_app.command("generate-demo")
def data_generate_demo(out_dir: Path = typer.Option(Path("data/demo"))) -> None:
    """US-103: (re)generate the versioned synthetic demo sample."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "scripts/generate_demo_data.py", "--out-dir", str(out_dir)], check=True
    )


@app.command()
def demo(
    conf_dir: Path = typer.Option(Path("conf")),
    reports_dir: Path = typer.Option(Path("reports")),
) -> None:
    """US-103 acceptance criterion: run the full pipeline on the demo sample
    end to end (ingest -> clean -> label -> split -> features F1-F4 ->
    M0/M1/M2 -> evaluate -> tables) and print a results summary.

    Deliberately scoped to the dependency-light models (floors, rules,
    pair-rarity) so `make demo` only requires the `classical` extra — the
    heavier deep/graph models are exercised by the full (non-demo) pipeline.
    """
    import time

    t0 = time.time()
    from omegaconf import OmegaConf

    dataset_cfg = OmegaConf.load(conf_dir / "dataset" / "demo.yaml")
    split_cfg = OmegaConf.load(conf_dir / "split" / "demo_temporal.yaml")

    auth_path = Path(str(dataset_cfg.auth_path))
    redteam_path = Path(str(dataset_cfg.redteam_path))
    if not auth_path.exists():
        console.print("[yellow]Demo data not found, generating it now...[/]")
        data_generate_demo(out_dir=auth_path.parent)

    console.print("[bold]1/7[/] Parsing and cleaning...")
    raw_auth = pl.scan_csv(
        auth_path,
        has_header=False,
        new_columns=list(dataset_cfg.auth_columns),
    ).with_columns(pl.col("time").cast(pl.Int64))
    typed_auth, drop_counts = clean_auth(raw_auth)

    raw_redteam = pl.scan_csv(
        redteam_path,
        has_header=False,
        new_columns=list(dataset_cfg.redteam_columns),
    ).with_columns(pl.col("time").cast(pl.Int64))
    typed_redteam, n_redteam_dupes = clean_redteam(raw_redteam)

    dq_report = data_quality_report(typed_auth, drop_counts)
    (reports_dir / "tables").mkdir(parents=True, exist_ok=True)
    (reports_dir / "data_quality.json").write_text(json.dumps(dq_report, indent=2, default=str))

    console.print("[bold]2/7[/] Labeling and campaign grouping...")
    campaigns = group_into_campaigns(typed_redteam, gap_hours=float(dataset_cfg.campaign_gap_hours))
    labeled = label_auth_events(typed_auth, typed_redteam)
    labeled = attach_campaign_id(labeled, campaigns)
    label_report = labeling_report(
        labeled, typed_redteam, campaigns.select("campaign_id").unique().collect().height
    )
    console.print(
        f"  {label_report.n_matched_in_auth} malicious events matched "
        f"({label_report.n_redteam_events} red-team rows, {n_redteam_dupes} duplicates removed), "
        f"positive rate {label_report.positive_rate:.2e}, {label_report.n_campaigns} campaigns."
    )
    campaign_summary(campaigns).write_csv(reports_dir / "tables" / "campaign_summary.csv")

    console.print("[bold]3/7[/] Temporal split with leakage guard...")
    split_config = TemporalSplitConfig.from_hydra(split_cfg)
    train = get_train_split(labeled, split_config)
    val = get_val_split(labeled, split_config)
    test = get_test_split(labeled, split_config)
    verify_temporal_order(train, val, test)
    console.print("  [green]No temporal leakage detected.[/]")

    console.print("[bold]4/7[/] Computing F1-F4 features...")
    # Everything fitted is fitted on `train` and only on `train` — the night
    # window and the F1 category frequencies alike. Refitting either one per
    # split is a leak, and it also makes the same category mean a different
    # number in train and in test.
    night_window = calibrate_night_window(train)
    frequency_encoding = fit_frequency_encoding(train)

    def featurize(frame: pl.LazyFrame) -> pl.LazyFrame:
        frame = compute_f1(frame, frequency_encoding)
        frame = compute_f2(frame)
        frame = compute_f3(frame)
        frame = compute_f4(frame, night_window)
        return frame

    train_feat = featurize(train).collect()
    val_feat = featurize(val).collect()
    test_feat = featurize(test).collect()

    console.print("[bold]5/7[/] Fitting models (M0, M1, M2, M3)...")
    models: list[object] = [
        RandomScorer(),
        AlwaysFailScorer(),
        PairRarityScorer(),
        PCAReconstructionScorer(F1_TO_F4_MODEL_FEATURES),
        IsolationForestScorer(F1_TO_F4_MODEL_FEATURES),
        ECODScorer(F1_TO_F4_MODEL_FEATURES),
        RulesScorer(),
    ]

    console.print("[bold]6/7[/] Scoring and evaluating on the test split...")
    y_test = test_feat["is_malicious"].to_numpy()

    results_table = Table(title="AuthBench demo — test-split results")
    results_table.add_column("Model")
    results_table.add_column("AUC-PR")
    for k in DEFAULT_BUDGETS:
        results_table.add_column(f"Campaign recall@{k}")

    all_rows = []
    curves = []
    for model in models:
        model.fit(train_feat.lazy())  # type: ignore[attr-defined]
        # M1 alone has a second, label-aware fitting step, and it must run
        # after `fit` (which re-derives the per-user R2 thresholds the rule
        # scores are built from) and against the validation split only.
        if isinstance(model, RulesScorer):
            model.calibrate_weights(val_feat.lazy(), val_feat["is_malicious"], n_trials=50)

        scores = model.score(test_feat.lazy())  # type: ignore[attr-defined]
        ap = auc_pr(y_test, scores.to_numpy())
        scored_frame = test_feat.with_columns(pl.Series("_score", scores))
        curve = compute_budget_curve(scored_frame, "_score", model.name)  # type: ignore[attr-defined]
        curves.append(curve)

        results_table.add_row(
            model.name,  # type: ignore[attr-defined]
            f"{ap:.4f}",
            *[f"{r:.2%}" for r in curve.campaign_recall],
        )
        all_rows.append(
            {
                "model": model.name,  # type: ignore[attr-defined]
                "auc_pr": ap,
                "budgets": curve.budgets,
                "campaign_recall": curve.campaign_recall,
            }
        )

    console.print("[bold]7/7[/] Writing report artifacts...")
    (reports_dir / "tables" / "demo_results.json").write_text(json.dumps(all_rows, indent=2))
    figure_path = plot_campaign_recall_vs_budget(
        curves,
        reports_dir / "figures" / CAMPAIGN_RECALL_FIGURE,
        subtitle=(
            f"Demo sample — {label_report.n_campaigns} campaigns, "
            f"{test_feat.height:,} test events. Not a LANL result."
        ),
    )
    console.print(results_table)
    console.print(f"Figure: {figure_path}")

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Demo pipeline completed in {elapsed:.1f}s.[/]")


if __name__ == "__main__":
    app()
