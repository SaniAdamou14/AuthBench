"""DVC `label_split_features` stage: raw Parquet -> labeled, split, feature-store Parquet.

This is the full-dataset counterpart of what `authbench.cli.demo` does inline
for the small demo sample — same library calls, driven by Hydra config
instead of hard-coded demo paths, and persisting a feature-store version
hash (US-115) so every model consumes exactly the same features.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig, OmegaConf

from authbench.features import FEATURE_STORE_COLUMNS
from authbench.features.event import FrequencyEncoding, compute_f1, fit_frequency_encoding
from authbench.features.history import (
    DEFAULT_DIVERSITY_PAIRS,
    DEFAULT_F2_ENTITIES,
    DEFAULT_F2_WINDOWS_HOURS,
    compute_f2,
)
from authbench.features.novelty import compute_f3
from authbench.features.temporal import NightWindow, calibrate_night_window, compute_f4
from authbench.label.redteam_join import (
    attach_campaign_id,
    campaign_summary,
    group_into_campaigns,
    label_auth_events,
)
from authbench.parse.clean import clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS
from authbench.split.temporal import (
    TemporalSplitConfig,
    verify_temporal_order,
)

logger = logging.getLogger(__name__)

CONF_DIR = Path(__file__).resolve().parents[3] / "conf"


def feature_store_version(features_cfg: DictConfig) -> str:
    """US-115: a hash of the feature configuration, so changing any feature
    parameter invalidates the cache and is traceable in every MLflow run.
    """
    canonical = json.dumps(OmegaConf.to_container(features_cfg, resolve=True), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def featurize(
    frame: pl.LazyFrame,
    features_cfg: DictConfig,
    night_window: NightWindow,
    frequency_encoding: FrequencyEncoding,
) -> pl.LazyFrame:
    if features_cfg.f1_event.enabled:
        frame = compute_f1(frame, frequency_encoding)
    if features_cfg.f2_history.enabled:
        frame = compute_f2(
            frame,
            # Only the entities, windows and diversity pairs anything reads —
            # four columns out of the fifty-four the full grid would produce.
            entities=DEFAULT_F2_ENTITIES,
            windows_hours=DEFAULT_F2_WINDOWS_HOURS,
            diversity_for=DEFAULT_DIVERSITY_PAIRS,
        )
    if features_cfg.f3_novelty.enabled:
        frame = compute_f3(frame)
    if features_cfg.f4_temporal.enabled:
        frame = compute_f4(frame, night_window)
    # F5/F6 require the optional graph/deep extras and the full trailing-window
    # dataset; wired in `authbench.features.graph`/`sequence` directly by
    # callers that opt in (spec: Should-priority, not on the critical path).
    return frame


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    interim_dir = Path(cfg.paths.interim_dir)
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir) / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)

    redteam_gz = raw_dir / "redteam.txt.gz"
    raw_redteam = pl.scan_csv(redteam_gz, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    typed_redteam, n_dupes = clean_redteam(raw_redteam)
    logger.info("Redteam: %d exact duplicates removed", n_dupes)

    campaigns = group_into_campaigns(typed_redteam, gap_hours=float(cfg.dataset.campaign_gap_hours))

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    campaign_summary(campaigns).write_csv(tables_dir / "campaign_summary.csv")

    split_config = TemporalSplitConfig.from_hydra(cfg.split)

    # One lazy frame per split, reading only that split's day partitions.
    #
    # The previous shape built a single graph — scan every day, join the labels,
    # attach campaigns, then filter three ways — and Polars re-executed the
    # whole thing for each terminal operation. `verify_temporal_order` alone
    # asks for four (train max, val min, val max, test min), so labeling 239
    # million rows happened four times before a single feature was computed,
    # and the stage died on a 532 MB allocation seconds after starting.
    #
    # Scanning by day partition instead pushes the filter down to the file
    # level: each split touches only its own files, and the join runs once per
    # split over that split's rows rather than repeatedly over all of them.
    warmup_days = int(cfg.features.get("history_warmup_days", 0))

    def raw_split(bounds: tuple[int, int], *, warmup: int = 0) -> pl.LazyFrame:
        """The split's auth events, *unlabeled*, plus `warmup` days before it.

        The warm-up days are read and featurised but never written: they exist
        so a split's first events have a past. Reaching backwards is always
        sound — those events precede the split and precede every later split —
        and reaching forwards is what the leakage guard forbids, which is why
        this only ever extends the *lower* bound.
        """
        low, high = bounds
        scan_low = max(0, low - warmup)
        paths = [interim_dir / "auth" / f"day={day}" for day in range(scan_low, high + 1)]
        present = [str(p / "*.parquet") for p in paths if p.exists()]
        if not present:
            raise FileNotFoundError(
                f"No converted day partitions for days {low}-{high} under {interim_dir / 'auth'}. "
                "A split that reaches outside the converted window has missing partitions, not "
                "empty ones — check params.yaml:to_parquet_days against conf/split/temporal.yaml."
            )
        return pl.scan_parquet(present)

    logger.info("Loading auth Parquet dataset from %s", interim_dir / "auth")
    # The splits proper, for verification and for the fitted quantities.
    train_raw = raw_split(split_config.train_days)
    val_raw = raw_split(split_config.val_days)
    test_raw = raw_split(split_config.test_days)
    if warmup_days:
        logger.info("Each split reads %d warm-up day(s) of history it will not write.", warmup_days)

    # Verified and fitted on the *unlabeled* frames, and that is the whole
    # point rather than an accident of ordering.
    #
    # None of these three reads a label: the leakage guard compares timestamps,
    # the night window buckets timestamps, and the frequency encoding counts
    # `auth_type` / `logon_type` / `auth_orientation`. Running them on the
    # labeled frames meant every one of their aggregations re-executed the
    # red-team join underneath — `verify_temporal_order` alone asks for four
    # (train max, val min, val max, test min) — so joining 137 million training
    # rows happened four times before a single feature existed, and the stage
    # died on a 44 MB allocation. On the raw frames Polars pushes the
    # projection down to a single column and answers from Parquet statistics.
    verify_temporal_order(train_raw, val_raw, test_raw)
    logger.info("Temporal split verified: no leakage between train/val/test.")

    # Both fitted quantities, and both fitted on the training split alone — a
    # night window or a category frequency taken from the full period is a leak
    # that no downstream check would catch.
    night_window = calibrate_night_window(train_raw)
    frequency_encoding = fit_frequency_encoding(train_raw)

    def labeled(frame: pl.LazyFrame) -> pl.LazyFrame:
        return attach_campaign_id(label_auth_events(frame, typed_redteam), campaigns)

    def checkpoint(frame: pl.LazyFrame, name: str) -> pl.LazyFrame:
        """Materialize a lazy frame to Parquet and rescan it.

        `featurize()` below joins many independently-computed feature branches
        back onto the same starting frame (one per F2 entity/window pair, plus
        F3's several sorts and F4's cumulative circular mean) — one `.join(...,
        on="event_id")` per branch. Polars' planner does not share the upstream
        work across sibling branches of a join graph: each branch re-derives
        the redteam label join and campaign-id assignment from the raw Parquet
        scan independently. Confirmed with `.explain()` on an 8-day train
        window: 32 independent full scans of the same ~134M-row base, each
        redoing the campaign-id sort/rank chain, before a single feature column
        existed — the same failure mode the module docstring above already
        diagnoses for `verify_temporal_order`, one level deeper.

        Checkpointing here breaks the graph exactly where it forked: every
        downstream branch now scans this small, already-labeled file instead
        of re-deriving it, at the cost of one extra Parquet round-trip per
        split. It changes nothing about the computed values — same join, same
        filter, just performed once.
        """
        tmp_path = interim_dir / f"_checkpoint_{name}.parquet"
        frame.sink_parquet(tmp_path, compression="zstd")
        return pl.scan_parquet(tmp_path)

    # Featurised over the warm-up window; written for the split's own days only.
    bounds = {
        "train": split_config.train_days,
        "val": split_config.val_days,
        "test": split_config.test_days,
    }
    warm = {
        name: checkpoint(labeled(raw_split(days, warmup=warmup_days)), name)
        for name, days in bounds.items()
    }

    version = feature_store_version(cfg.features)
    version_dir = processed_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "feature_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg.features, resolve=True), indent=2)
    )

    for split_name, frame in warm.items():
        out = featurize(frame, cfg.features, night_window, frequency_encoding)
        # Warm-up rows are dropped *after* featurisation: they were read so the
        # split's own first events would have a past, not to be evaluated on.
        out = out.filter(pl.col("day") >= bounds[split_name][0])
        # Projected down to what downstream stages read. Also the point at
        # which a missing column becomes an error here, at the boundary, rather
        # than a `ColumnNotFoundError` from inside a model's `fit`.
        available = set(out.collect_schema().names())
        missing = [c for c in FEATURE_STORE_COLUMNS if c not in available]
        if missing:
            raise KeyError(
                f"Feature pipeline did not produce {missing} for the {split_name} split. "
                "Either a feature family is disabled in conf/features/ while something "
                "downstream still needs its output, or FEATURE_STORE_COLUMNS is stale."
            )
        out = out.select(FEATURE_STORE_COLUMNS)

        out_path = version_dir / f"{split_name}.parquet"
        out.sink_parquet(out_path, compression="zstd")
        logger.info("Wrote %s features to %s", split_name, out_path)

        checkpoint_path = interim_dir / f"_checkpoint_{split_name}.parquet"
        checkpoint_path.unlink(missing_ok=True)

    logger.info("Feature store version: %s", version)


if __name__ == "__main__":
    main()
