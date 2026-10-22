"""DVC `label_split_features` stage: day-partitioned Parquet -> labelled, split, feature-store Parquet.

Days are read, labelled and featurised one at a time (`features.incremental`),
so memory is bounded by a day and its predecessor whatever the length of the
run. Each partition is written as one Parquet file per day under
`<feature store>/<split>/`, and the store's version is a hash of the feature
configuration (US-115).

Partitions are half-open time ranges, given in days or in hours. History is
either `window` (each partition is featurised from its own first day minus
`history_warmup_days`, the published configuration) or `cumulative` (one pass
from day 0, so every partition reads all earlier events, the protocol of
Larroche (2026) and of reviewer 3).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
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
from authbench.features.incremental import FeatureState, featurize_day
from authbench.features.novelty import compute_f3
from authbench.features.temporal import NightWindow, calibrate_night_window, compute_f4
from authbench.label.redteam_join import (
    attach_campaign_id,
    campaign_summary,
    group_into_campaigns,
    label_auth_events,
    split_label_summary,
)
from authbench.parse.clean import clean_redteam
from authbench.parse.schema import RAW_REDTEAM_COLUMNS, SECONDS_PER_DAY

logger = logging.getLogger(__name__)

CONF_DIR = Path(__file__).resolve().parents[3] / "conf"
UNIVERSES = ("all", "remote_logon", "remote_logon_user")
SPLITS = ("train", "val", "test")


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
    """F1-F4 over a whole window at once: the reference `featurize_day` is tested against."""
    if features_cfg.f1_event.enabled:
        frame = compute_f1(frame, frequency_encoding)
    if features_cfg.f2_history.enabled:
        frame = compute_f2(
            frame,
            entities=DEFAULT_F2_ENTITIES,
            windows_hours=DEFAULT_F2_WINDOWS_HOURS,
            diversity_for=DEFAULT_DIVERSITY_PAIRS,
        )
    if features_cfg.f3_novelty.enabled:
        frame = compute_f3(frame)
    if features_cfg.f4_temporal.enabled:
        frame = compute_f4(frame, night_window)
    return frame


def universe_filter(universe: str) -> pl.Expr:
    """Events kept: all, remote LogOn (Larroche, 2026), or remote LogOn to a user account."""
    if universe not in UNIVERSES:
        raise ValueError(f"Unknown universe {universe!r}; expected one of {UNIVERSES}.")
    if universe == "all":
        return pl.lit(True)
    remote_logon = (pl.col("auth_orientation").cast(pl.Utf8) == "LogOn") & (
        pl.col("src_computer").cast(pl.Utf8) != pl.col("dst_computer").cast(pl.Utf8)
    )
    if universe == "remote_logon":
        return remote_logon
    return remote_logon & ~pl.col("dst_user_is_machine")


def split_ranges(split_cfg: DictConfig) -> dict[str, tuple[int, int]]:
    """Half-open [start, end) seconds per partition, from `<name>_hours` or inclusive `<name>_days`."""
    ranges = {}
    for name in SPLITS:
        if f"{name}_hours" in split_cfg:
            low, high = (int(h) for h in split_cfg[f"{name}_hours"])
            ranges[name] = (low * 3600, high * 3600)
        elif f"{name}_days" in split_cfg:
            low, high = (int(d) for d in split_cfg[f"{name}_days"])
            ranges[name] = (low * SECONDS_PER_DAY, (high + 1) * SECONDS_PER_DAY)
    if "train" not in ranges or "test" not in ranges:
        raise ValueError(f"A split needs train and test ranges, got {sorted(ranges)}.")
    ordered = list(ranges.values())
    if any(low >= high for low, high in ordered) or any(
        a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError(f"Partitions are empty, overlap or are out of temporal order: {ranges}.")
    return ranges


def days_of(start: int, end: int) -> range:
    """Days touched by the half-open range [start, end) in seconds."""
    return range(start // SECONDS_PER_DAY, (end - 1) // SECONDS_PER_DAY + 1)


@dataclass(frozen=True)
class StoreInputs:
    """Everything `build_store` reads: unlabelled days by number, labels, and the training fits."""

    read_day: Callable[[int], pl.DataFrame | None]
    redteam: pl.LazyFrame
    campaigns: pl.LazyFrame
    night_window: NightWindow
    frequency_encoding: FrequencyEncoding


def build_store(
    inputs: StoreInputs,
    ranges: dict[str, tuple[int, int]],
    out_dir: Path,
    *,
    history: str = "window",
    warmup_days: int = 0,
) -> dict[str, object]:
    """Featurise each partition day by day into `<out_dir>/<split>/day=NNN.parquet`; return split summaries."""
    if history not in ("window", "cumulative"):
        raise ValueError(f"history must be 'window' or 'cumulative', not {history!r}.")
    for name in SPLITS:
        # Day files left by an earlier split configuration would otherwise be scanned with the new ones.
        shutil.rmtree(out_dir / name, ignore_errors=True)

    def write(features: pl.DataFrame, day: int) -> None:
        for name, (start, end) in ranges.items():
            rows = features.filter(pl.col("time").is_between(start, end, closed="left"))
            if rows.height:
                (out_dir / name).mkdir(parents=True, exist_ok=True)
                rows.select(FEATURE_STORE_COLUMNS).write_parquet(
                    out_dir / name / f"day={day:03d}.parquet", compression="zstd"
                )

    def run(days: range, first_written: int) -> None:
        state = FeatureState()
        for day in days:
            frame = inputs.read_day(day)
            if frame is None:
                raise FileNotFoundError(
                    f"Day {day} is not converted; check params.yaml:to_parquet_days against conf/split/."
                )
            if not frame.height:
                continue
            frame = attach_campaign_id(
                label_auth_events(frame.lazy(), inputs.redteam), inputs.campaigns
            ).collect()
            features, state = featurize_day(
                frame, state, inputs.night_window, inputs.frequency_encoding
            )
            if day >= first_written:
                write(features, day)
            logger.info("Featurised day %d: %d events.", day, frame.height)

    if history == "cumulative":
        first = min(start for start, _ in ranges.values()) // SECONDS_PER_DAY
        last = days_of(*max(ranges.values())).stop
        run(range(0, last), first)
    else:
        for start, end in ranges.values():
            days = days_of(start, end)
            run(range(max(0, days.start - warmup_days), days.stop), days.start)

    return {
        name: split_label_summary(pl.scan_parquet(out_dir / name / "*.parquet"))
        for name in ranges
        if (out_dir / name).exists()
    }


@hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    auth_dir = Path(cfg.paths.interim_dir) / "auth"
    tables_dir = Path(cfg.paths.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_redteam = pl.scan_csv(
        Path(cfg.paths.raw_dir) / "redteam.txt.gz",
        has_header=False,
        new_columns=RAW_REDTEAM_COLUMNS,
    )
    typed_redteam, n_dupes = clean_redteam(raw_redteam)
    logger.info("Redteam: %d exact duplicates removed", n_dupes)
    campaigns = group_into_campaigns(typed_redteam, gap_hours=float(cfg.dataset.campaign_gap_hours))
    campaign_summary(campaigns).write_csv(tables_dir / "campaign_summary.csv")

    keep = universe_filter(str(cfg.features.get("universe", "all")))
    ranges = split_ranges(cfg.split)

    def scan_day(day: int) -> pl.LazyFrame | None:
        path = auth_dir / f"day={day}"
        return pl.scan_parquet(path / "*.parquet").filter(keep) if path.exists() else None

    def read_day(day: int) -> pl.DataFrame | None:
        frame = scan_day(day)
        return None if frame is None else frame.collect()

    train_start, train_end = ranges["train"]
    train_days = [scan_day(d) for d in days_of(train_start, train_end)]
    if any(frame is None for frame in train_days):
        raise FileNotFoundError(
            f"Training days {list(days_of(train_start, train_end))} are not all converted."
        )
    train_raw = pl.concat([f for f in train_days if f is not None]).filter(
        pl.col("time").is_between(train_start, train_end, closed="left")
    )
    # Fitted on the training range alone: a night window or a category frequency from later days is a leak.
    inputs = StoreInputs(
        read_day=read_day,
        redteam=typed_redteam,
        campaigns=campaigns,
        night_window=calibrate_night_window(train_raw),
        frequency_encoding=fit_frequency_encoding(train_raw),
    )

    version = feature_store_version(cfg.features)
    version_dir = Path(cfg.paths.processed_dir) / "features" / version
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "feature_config.json").write_text(
        json.dumps(OmegaConf.to_container(cfg.features, resolve=True), indent=2)
    )
    summaries = build_store(
        inputs,
        ranges,
        version_dir,
        history=str(cfg.split.get("history", "window")),
        warmup_days=int(cfg.features.get("history_warmup_days", 0)),
    )
    (tables_dir / "split_summary.json").write_text(json.dumps(summaries, indent=2))
    logger.info("Feature store version: %s", version)


if __name__ == "__main__":
    main()
