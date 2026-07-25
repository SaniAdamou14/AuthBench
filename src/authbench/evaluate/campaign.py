"""Time-to-detection per campaign (US-127)."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from authbench.evaluate.budget import alerts_at_budget


@dataclass
class TimeToDetectionResult:
    model_name: str
    budget: int
    detected_delay_seconds: list[float] = field(default_factory=list)
    n_never_detected: int = 0
    n_total_campaigns: int = 0

    @property
    def n_detected(self) -> int:
        return len(self.detected_delay_seconds)


def time_to_detection(
    frame: pl.DataFrame, score_col: str, k: int, model_name: str
) -> TimeToDetectionResult:
    """Delay between a campaign's first event and the first alert that
    concerns it, at alert budget `k`.

    Campaigns never detected at this budget are counted in
    `n_never_detected` and excluded from `detected_delay_seconds` — never
    silently dropped from the denominator (US-127 acceptance criterion).
    """
    campaign_events = frame.filter(pl.col("campaign_id").is_not_null())
    first_event_time = campaign_events.group_by("campaign_id").agg(
        pl.col("time").min().alias("first_event_time")
    )

    alerts = alerts_at_budget(frame, score_col, k).filter(pl.col("campaign_id").is_not_null())
    first_alert_time = alerts.group_by("campaign_id").agg(
        pl.col("time").min().alias("first_alert_time")
    )

    joined = first_event_time.join(first_alert_time, on="campaign_id", how="left")
    detected = joined.filter(pl.col("first_alert_time").is_not_null())
    never_detected = joined.filter(pl.col("first_alert_time").is_null())

    delays = (detected["first_alert_time"] - detected["first_event_time"]).to_list()

    return TimeToDetectionResult(
        model_name=model_name,
        budget=k,
        detected_delay_seconds=delays,
        n_never_detected=never_detected.height,
        n_total_campaigns=joined.height,
    )
