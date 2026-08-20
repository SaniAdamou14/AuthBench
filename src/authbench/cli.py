"""AuthBench command-line interface.

Every subcommand is a thin wrapper around `authbench.*` library code — the
CLI itself holds no business logic, so everything here is equally usable
from a notebook or a DVC stage.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import polars as pl
import typer
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

from authbench.evaluate.budget import DEFAULT_BUDGETS
from authbench.evaluate.metrics import ROC_AUC_WARNING, auc_pr
from authbench.evaluate.plots import CAMPAIGN_RECALL_FIGURE, plot_campaign_recall_vs_budget
from authbench.evaluate.stats_tests import (
    MIN_CAMPAIGNS_FOR_SIGNIFICANCE,
    paired_campaign_bootstrap,
    render_comparison_sentence,
)
from authbench.evaluate.summary import EVAL_COLUMNS, evaluate_model, score_column
from authbench.features import MODEL_FEATURE_COLUMNS
from authbench.features.event import compute_f1, fit_frequency_encoding
from authbench.features.history import compute_f2
from authbench.features.novelty import compute_f3
from authbench.features.temporal import calibrate_night_window, compute_f4
from authbench.ingest.budget import (
    available_disk_bytes,
    available_memory_bytes,
    lanl_budget,
    max_events_for_disk,
    max_events_for_memory,
    peak_rss_bytes,
    total_disk_bytes,
)
from authbench.ingest.download import (
    download_with_resume,
    fetch_lanl_fence_token,
    free_space_bytes,
    lanl_file_url,
)
from authbench.ingest.to_parquet import (
    estimate_parquet_bytes,
    to_parquet_partitioned,
    verify_row_count,
)
from authbench.label.redteam_join import (
    attach_campaign_id,
    campaign_summary,
    group_into_campaigns,
    label_auth_events,
    labeling_report,
)
from authbench.models.classical import ECODScorer, IsolationForestScorer
from authbench.models.floors import AlwaysFailScorer, RandomScorer
from authbench.models.rules import RulesScorer, rule_discrimination
from authbench.models.stats import PairRarityScorer, PCAReconstructionScorer
from authbench.parse.clean import clean_auth, clean_redteam, data_quality_report
from authbench.pipeline.build_features import CONF_DIR
from authbench.split.temporal import (
    TemporalSplitConfig,
    get_test_split,
    get_train_split,
    get_val_split,
    verify_temporal_order,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("authbench.cli")

# A legacy Windows console reports cp1252, and Python then raises
# UnicodeEncodeError on the first character it cannot map — which for this
# CLI means a full crash at the very end of a completed run, on a report
# sentence containing "Δ". CI is Linux/UTF-8 and never sees it, so the whole
# class of failure is invisible until someone runs `make demo` on Windows.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="AuthBench — leak-free, alert-budget-constrained auth-log anomaly benchmark."
)
data_app = typer.Typer(help="Data acquisition and conversion.")
app.add_typer(data_app, name="data")

console = Console()

# The demo's own evaluation knobs. The full pipeline reads these from
# `conf/eval/default.yaml`; the demo is a fixed, self-contained smoke test
# with a five-minute CI ceiling (NFR-04), so it pins smaller values rather
# than pulling in the Hydra config it does not otherwise need.
# Enough to resolve a p-value below Holm's strictest threshold for the demo
# catalog's 7 models (21 pairs → 0.05/21 = 0.00238; 2/(1000+1) = 0.00200).
# Below that, no comparison could ever be significant however good a model
# was — see `stats_tests.minimum_resamples_for_family`.
DEMO_BOOTSTRAP_RESAMPLES = 1000
DEMO_FPR_TARGETS = [1.0e-4, 1.0e-3]


def load_dataset_config(dataset: str) -> DictConfig:
    """Load `conf/dataset/<dataset>.yaml`, from the working directory if it is
    a checkout and from the installed package's `conf/` otherwise.

    The previous `OmegaConf.load(f"conf/dataset/{dataset}.yaml")` silently
    depended on the process's working directory, so `authbench data download`
    worked from the repository root and failed everywhere else with a bare
    `FileNotFoundError` naming a relative path.
    """
    candidates = [
        Path("conf") / "dataset" / f"{dataset}.yaml",
        CONF_DIR / "dataset" / f"{dataset}.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            loaded = OmegaConf.load(candidate)
            assert isinstance(loaded, DictConfig)
            return loaded
    raise typer.BadParameter(
        f"No config for dataset {dataset!r}. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Available: "
        + ", ".join(sorted(p.stem for p in (CONF_DIR / "dataset").glob("*.yaml"))),
        param_hint="--dataset",
    )


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
    if not email:
        raise typer.BadParameter(
            "LANL's data-use form requires an email address. Pass --email, or set "
            "AUTHBENCH_LANL_EMAIL in your environment (what `dvc repro` expects).",
            param_hint="--email",
        )

    cfg = load_dataset_config(dataset)
    console.print(f"Destination {out_dir}: {free_space_bytes(out_dir) / 1e9:.1f} GB free.")

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
    dataset: str = typer.Option(
        "lanl", help="Dataset whose conf/dataset/<name>.yaml supplies the expected row count."
    ),
    expected_rows: int | None = typer.Option(
        None,
        help="Abort if the converted row count mismatches. Defaults to the dataset config's "
        "expected_rows.auth; pass 0 to skip the check entirely.",
    ),
    block_bytes: int | None = typer.Option(
        None, help="Decompressed bytes held in memory per block. Lower it on a small machine."
    ),
) -> None:
    """US-102: stream raw text into day-partitioned, ZSTD-compressed Parquet.

    The row-count target comes from `conf/dataset/<dataset>.yaml` rather than
    from the caller. A published constant repeated in `dvc.yaml`, in the
    README and in a shell history is a constant that will eventually
    disagree with itself, and the whole point of the check is that it is the
    published figure (US-102).
    """
    if expected_rows is None:
        cfg = load_dataset_config(dataset)
        configured = cfg.get("expected_rows", {}).get("auth")
        expected_rows = int(configured) if configured is not None else None

    console.print(
        f"Source {src.name}: {src.stat().st_size / 1e9:.2f} GB on disk. "
        f"Estimated Parquet output ~{estimate_parquet_bytes(src) / 1e9:.1f} GB, "
        f"{free_space_bytes(out_dir) / 1e9:.1f} GB free on the destination volume."
    )

    kwargs = {} if block_bytes is None else {"block_bytes": block_bytes}
    report = to_parquet_partitioned(src, out_dir, **kwargs)  # type: ignore[arg-type]
    console.print(
        f"Converted {report.n_rows_out:,}/{report.n_rows_in:,} rows in {report.n_batches} "
        f"blocks, peak RSS {report.peak_rss_bytes / 1e9:.2f} GB "
        f"(dropped {report.drop_counts.total:,}: "
        f"{report.drop_counts.null_time:,} null time, "
        f"{report.drop_counts.malformed_user_domain:,} malformed user@domain)."
    )
    if expected_rows:
        verify_row_count(report, expected_rows)
        console.print(f"[green]Row count verified against {expected_rows:,}.[/]")
    else:
        console.print("[yellow]Row-count verification skipped.[/]")


@app.command()
def preflight(
    data_dir: Path = typer.Option(Path("data"), help="Where the pipeline will write its data."),
    dataset: str = typer.Option("lanl", help="Dataset to budget for."),
    events: int | None = typer.Option(
        None, help="Budget for this many auth events instead of the dataset's published total."
    ),
) -> None:
    """Disk, memory and dependency budget for a full run — before downloading.

    Prints what each stage costs on this machine and exits non-zero if it does
    not fit, together with the event count that would. A 1.05-billion-event
    conversion has no business discovering it is out of disk five hours in.
    """
    import importlib.util

    cfg = load_dataset_config(dataset)
    n_events = events or int(cfg.get("expected_rows", {}).get("auth") or 0)
    if not n_events:
        raise typer.BadParameter(
            f"conf/dataset/{dataset}.yaml declares no expected_rows.auth; pass --events.",
            param_hint="--events",
        )

    budget = lanl_budget(n_events)
    disk_needed = total_disk_bytes(budget)
    rss_needed = peak_rss_bytes(budget)
    disk_free = available_disk_bytes(data_dir)
    ram_total = available_memory_bytes()

    table = Table(title=f"Budget for {dataset} — {n_events:,} auth events")
    table.add_column("Stage")
    table.add_column("Disk added", justify="right")
    table.add_column("Peak RAM", justify="right")
    table.add_column("Note")
    for stage in budget:
        table.add_row(
            stage.stage,
            f"{stage.disk_bytes / 1e9:.1f} GB",
            f"{stage.peak_rss_bytes / 1e9:.1f} GB",
            stage.note,
        )
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{disk_needed / 1e9:.1f} GB[/]",
        f"[bold]{rss_needed / 1e9:.1f} GB[/]",
        "disk is cumulative, RAM is the largest single stage",
    )
    console.print(table)

    console.print(
        f"\nThis machine: {disk_free / 1e9:.1f} GB free under {data_dir.resolve()}, "
        f"{ram_total / 1e9:.1f} GB RAM."
    )

    missing = [
        name
        for module, name in [
            ("dvc", "dvc (extra: tracking)"),
            ("mlflow", "mlflow (extra: tracking, optional)"),
            ("pyod", "pyod (extra: classical)"),
        ]
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        console.print(
            f"[yellow]Not installed: {', '.join(missing)}. `make install-lanl` covers them.[/]"
        )

    problems: list[str] = []
    if disk_free < disk_needed:
        fits = max_events_for_disk(disk_free)
        problems.append(
            f"Disk: need {disk_needed / 1e9:.1f} GB, have {disk_free / 1e9:.1f} GB. "
            f"At this free space the pipeline fits about {fits:,} events "
            f"({100 * fits / n_events:.1f}% of {dataset})."
        )
    if ram_total and ram_total < rss_needed:
        fits = max_events_for_memory(ram_total)
        problems.append(
            f"Memory: train_eval needs about {rss_needed / 1e9:.1f} GB for the design matrix, "
            f"this machine has {ram_total / 1e9:.1f} GB. That caps the run at roughly "
            f"{fits:,} events."
        )

    if problems:
        console.print("\n[bold red]This run does not fit on this machine.[/]")
        for problem in problems:
            console.print(f"  - {problem}")
        console.print(
            "\nOptions: free space / use another drive (every path is a CLI flag or a "
            "conf/ value), or run a documented subset and say so in the report. "
            "See docs/scaling.md."
        )
        raise typer.Exit(code=1)

    console.print("\n[bold green]Budget fits.[/]")


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
    reports_dir: Path = typer.Option(
        Path("reports/demo"),
        help="Where the demo writes its artifacts. The default is versioned in git — "
        "see reports/demo/RUN.md.",
    ),
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
        PCAReconstructionScorer(MODEL_FEATURE_COLUMNS),
        IsolationForestScorer(MODEL_FEATURE_COLUMNS),
        ECODScorer(MODEL_FEATURE_COLUMNS),
        RulesScorer(),
    ]

    console.print("[bold]6/7[/] Scoring and evaluating on the test split...")
    model_names: list[str] = [m.name for m in models]  # type: ignore[attr-defined]
    scored = test_feat.select(EVAL_COLUMNS)
    for model in models:
        model.fit(train_feat.lazy())  # type: ignore[attr-defined]
        # M1 alone has a second, label-aware fitting step, and it must run
        # after `fit` (which re-derives the per-user R2 thresholds the rule
        # scores are built from) and against the validation split only.
        if isinstance(model, RulesScorer):
            model.calibrate_weights(val_feat.lazy(), val_feat["is_malicious"], n_trials=50)

        scores = model.score(test_feat.lazy())  # type: ignore[attr-defined]
        scored = scored.with_columns(pl.Series(score_column(model.name), scores))  # type: ignore[attr-defined]

        if isinstance(model, RulesScorer):
            rule_discrimination(model, test_feat).write_csv(
                reports_dir / "tables" / "rule_discrimination.csv"
            )

    # One campaign-stratified resampling pass serves every model: the
    # per-model CIs and the pairwise differences then come from the same
    # sampling distribution. Fewer resamples than the full pipeline's 1000 —
    # the demo is a smoke test under a five-minute CI ceiling (NFR-04).
    bootstrap = paired_campaign_bootstrap(
        scored,
        lambda frame, col: auc_pr(frame["is_malicious"].to_numpy(), frame[col].to_numpy()),
        {name: score_column(name) for name in model_names},
        n_resamples=DEMO_BOOTSTRAP_RESAMPLES,
        seed=42,
    )

    # Two tables, because they are two different claims. The operational one
    # is what a SOC would live with; the literature-comparable one exists so
    # these numbers can sit next to published ones — and, on this sample,
    # to show how far apart the two registers can drift.
    operational = Table(title="Operational — what an analyst at this budget actually gets")
    operational.add_column("Model")
    operational.add_column("AUC-PR [95% CI]")
    for k in DEFAULT_BUDGETS:
        operational.add_column(f"Camp.rec@{k}", justify="right")
    operational.add_column("TTD@100", justify="right")

    comparable = Table(title="Literature-comparable — reported for placement, not for ranking")
    comparable.add_column("Model")
    comparable.add_column("ROC-AUC", justify="right")
    for k in DEFAULT_BUDGETS:
        comparable.add_column(f"P@{k}", justify="right")
    for fpr in DEMO_FPR_TARGETS:
        comparable.add_column(f"Rec@FPR{fpr:g}", justify="right")

    all_rows = []
    curves = []
    for name in model_names:
        evaluation = evaluate_model(
            scored,
            score_column(name),
            name,
            budgets=DEFAULT_BUDGETS,
            fpr_targets=DEMO_FPR_TARGETS,
            # Emitted once, under the table it applies to, rather than seven
            # times into the middle of the run.
            warn_roc_auc=False,
            auc_pr_ci=bootstrap.ci(name),
        )
        curves.append(evaluation.curve)
        all_rows.append(evaluation.to_dict())

        ci = bootstrap.ci(name)
        ttd = next(t for t in evaluation.time_to_detection if t.budget == 100)
        operational.add_row(
            name,
            f"{evaluation.auc_pr:.4f} [{ci.ci_low:.4f}, {ci.ci_high:.4f}]",
            *[f"{r:.0%}" for r in evaluation.curve.campaign_recall],
            (
                f"{ttd.median_delay_seconds / 60:.0f}min"
                if ttd.median_delay_seconds is not None
                else f"none {ttd.n_never_detected}/{ttd.n_total_campaigns}"
            ),
        )
        comparable.add_row(
            name,
            f"{evaluation.roc_auc:.4f}" if evaluation.roc_auc is not None else "—",
            *[f"{evaluation.precision_at_k_global[k]:.3f}" for k in DEFAULT_BUDGETS],
            *[f"{evaluation.recall_at_fixed_fpr.get(fpr, 0.0):.3f}" for fpr in DEMO_FPR_TARGETS],
        )

    comparisons = bootstrap.comparisons("auc_pr", alpha=0.05)

    console.print("[bold]7/7[/] Writing report artifacts...")
    (reports_dir / "tables" / "demo_results.json").write_text(json.dumps(all_rows, indent=2))
    (reports_dir / "tables" / "pairwise_comparisons.json").write_text(
        json.dumps([c.to_dict() for c in comparisons], indent=2)
    )
    # The figure plots campaign recall over the campaigns present *in the
    # test split*, so that is the count the subtitle has to carry. Printing
    # the demo's total (3, one per partition) next to a curve computed over
    # 1 would misstate the denominator of every point on it.
    n_test_campaigns = scored.filter(pl.col("campaign_id").is_not_null())["campaign_id"].n_unique()
    figure_path = plot_campaign_recall_vs_budget(
        curves,
        reports_dir / "figures" / CAMPAIGN_RECALL_FIGURE,
        subtitle=(
            f"Demo sample — {n_test_campaigns} campaign(s) in the test split of "
            f"{label_report.n_campaigns} total, {test_feat.height:,} test events. "
            "Not a LANL result."
        ),
    )
    console.print(operational)
    console.print(comparable)
    console.print(f"[yellow]{ROC_AUC_WARNING}[/]")
    console.print(f"Figure: {figure_path}")

    # US-128: the comparison is only allowed to say "outperforms" on the
    # significant branch, and on three campaigns almost nothing is. Printing
    # the significant ones plus a count of the rest is the honest summary.
    console.print("\n[bold]Pairwise AUC-PR comparisons[/] (Holm-Bonferroni, alpha=0.05):")
    significant = [c for c in comparisons if c.significant]
    for comparison in significant:
        console.print(f"  {render_comparison_sentence(comparison)}")
    n_undecided = len(comparisons) - len(significant)
    console.print(
        f"  {len(significant)}/{len(comparisons)} pairs significant"
        + (
            f"; the remaining {n_undecided} are indistinguishable at this sample size."
            if n_undecided
            else "."
        )
    )
    if bootstrap.n_campaign_blocks < MIN_CAMPAIGNS_FOR_SIGNIFICANCE:
        # Distinguish "we looked and found no difference" from "this split
        # cannot answer the question" — they print the same way otherwise, and
        # only one of them is a result.
        console.print(
            f"  [yellow]Not that they are close: with {bootstrap.n_campaign_blocks} campaign(s) "
            "in the test split a campaign-stratified bootstrap has no campaign-level variation "
            "to resample, so significance is not estimable at all here. The AUC-PR point "
            "estimates and their intervals stand; the verdict is withheld.[/]"
        )
    if bootstrap.n_degenerate_discarded:
        # Not noise: it says the test split is thin on campaigns, which is
        # the thing that most limits what any of these intervals can mean.
        console.print(
            f"  ({bootstrap.n_degenerate_discarded} resamples drew no positive at all and "
            f"were redrawn — {n_test_campaigns} campaign(s) in the test split.)"
        )

    elapsed = time.time() - t0
    console.print(f"\n[bold green]Demo pipeline completed in {elapsed:.1f}s.[/]")


if __name__ == "__main__":
    app()
