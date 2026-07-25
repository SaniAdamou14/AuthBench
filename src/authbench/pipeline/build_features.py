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

from authbench.features.event import compute_f1
from authbench.features.history import compute_f2
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
    get_test_split,
    get_train_split,
    get_val_split,
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
    frame: pl.LazyFrame, features_cfg: DictConfig, night_window: NightWindow
) -> pl.LazyFrame:
    if features_cfg.f1_event.enabled:
        frame = compute_f1(frame)
    if features_cfg.f2_history.enabled:
        frame = compute_f2(frame, windows_hours=list(features_cfg.f2_history.windows_hours))
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

    logger.info("Loading auth Parquet dataset from %s", interim_dir / "auth")
    auth = pl.scan_parquet(interim_dir / "auth" / "**" / "*.parquet")

    redteam_gz = raw_dir / "redteam.txt.gz"
    raw_redteam = pl.scan_csv(redteam_gz, has_header=False, new_columns=RAW_REDTEAM_COLUMNS)
    typed_redteam, n_dupes = clean_redteam(raw_redteam)
    logger.info("Redteam: %d exact duplicates removed", n_dupes)

    campaigns = group_into_campaigns(typed_redteam, gap_hours=float(cfg.dataset.campaign_gap_hours))
    labeled = label_auth_events(auth, typed_redteam)
    labeled = attach_campaign_id(labeled, campaigns)

    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    campaign_summary(campaigns).write_csv(tables_dir / "campaign_summary.csv")

    split_config = TemporalSplitConfig.from_hydra(cfg.split)
    train = get_train_split(labeled, split_config)
    val = get_val_split(labeled, split_config)
    test = get_test_split(labeled, split_config)
    verify_temporal_order(train, val, test)
    logger.info("Temporal split verified: no leakage between train/val/test.")

    night_window = calibrate_night_window(train)

    version = feature_store_version(cfg.features)
    version_dir = processed_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "feature_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg.features, resolve=True), indent=2)
    )

    for split_name, split_frame in [("train", train), ("val", val), ("test", test)]:
        out = featurize(split_frame, cfg.features, night_window)
        out_path = version_dir / f"{split_name}.parquet"
        out.sink_parquet(out_path, compression="zstd")
        logger.info("Wrote %s features to %s", split_name, out_path)

    logger.info("Feature store version: %s", version)


if __name__ == "__main__":
    main()
