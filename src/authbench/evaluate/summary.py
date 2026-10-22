"""The full per-model evaluation record (US-125 to US-127).

One place assembles every metric a model is reported on, so `authbench demo`
and the DVC `train_eval` stage cannot drift into reporting different things.

Metrics come in two registers and the distinction is deliberate:

- **Operational** — recall at a daily alert budget, campaign recall, and
  time-to-detection. These answer "at a budget an analyst could actually
  work, does this catch the attack, and how fast?".
- **Literature-comparable** — ROC-AUC, global precision@k, recall at a fixed
  FPR. These exist so results can be placed next to published numbers, and
  each carries the caveat that makes it comparable rather than meaningful:
  at a ~1e-7 positive rate they stay flattering for models no SOC could use.

`to_dict` keeps them under separate keys in the JSON for the same reason.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

from authbench.evaluate import statistics
from authbench.evaluate.budget import BudgetCurve
from authbench.evaluate.metrics import ROC_AUC_WARNING
from authbench.evaluate.stats_tests import BootstrapResult, PairedBootstrap, PairwiseComparison

logger = logging.getLogger(__name__)

# The only columns evaluation reads from a scored day (NFR-01).
EVAL_COLUMNS = [
    "event_id",
    "time",
    "day",
    "is_malicious",
    "campaign_id",
    "src_user",
    "src_computer",
    "dst_computer",
]


def score_column(model_name: str) -> str:
    """Where a model's scores live on the shared evaluation frame.

    Every model's scores sit on one frame, under its own column, so a single
    block-bootstrap resampling pass can serve all of them at once —
    which is also what makes the pairwise differences paired.
    """
    return f"_score__{model_name}"


@dataclass
class TimeToDetectionSummary:
    """`evaluate.campaign.TimeToDetectionResult`, reduced to what a table shows.

    Campaigns never detected at this budget keep their own count and stay out
    of the delay statistics — never folded into the denominator as if they
    had been detected instantly (US-127).
    """

    budget: int
    n_total_campaigns: int
    n_detected: int
    n_never_detected: int
    median_delay_seconds: float | None
    mean_delay_seconds: float | None
    max_delay_seconds: float | None

    @classmethod
    def from_statistics(cls, stats: statistics.ModelStatistics, k: int) -> TimeToDetectionSummary:
        result = statistics.time_to_detection(stats, k)
        delays = np.asarray(result.detected_delay_seconds, dtype=np.float64)
        has_any = delays.size > 0
        return cls(
            budget=k,
            n_total_campaigns=result.n_total_campaigns,
            n_detected=result.n_detected,
            n_never_detected=result.n_never_detected,
            median_delay_seconds=float(np.median(delays)) if has_any else None,
            mean_delay_seconds=float(np.mean(delays)) if has_any else None,
            max_delay_seconds=float(np.max(delays)) if has_any else None,
        )


@dataclass(frozen=True)
class ScoreResolution:
    """How finely a model's score separates events, on the two sides where ties matter.

    ROC-AUC credits a (positive, benign) pair on the same score one half, so a coarse
    score is pulled toward 0.5 by arithmetic rather than by its operating point;
    `tied_pair_fraction` is the share of pairs that credit applies to. The alert queue
    breaks ties within a day; the two per-day fields say how much of a day one score
    value can hold. All three are exact and computed one day at a time.
    """

    n_events: int
    tied_pair_fraction: float | None
    min_distinct_scores_in_a_day: int
    largest_tie_fraction_in_a_day: float

    @classmethod
    def from_statistics(cls, stats: statistics.ModelStatistics) -> ScoreResolution:
        return cls(
            n_events=stats.n_events,
            tied_pair_fraction=statistics.tied_pair_fraction(stats),
            min_distinct_scores_in_a_day=stats.min_distinct_scores_in_a_day,
            largest_tie_fraction_in_a_day=stats.largest_tie_fraction_in_a_day,
        )


@dataclass
class ModelEvaluation:
    model_name: str
    auc_pr: float
    curve: BudgetCurve
    time_to_detection: list[TimeToDetectionSummary]
    auc_pr_ci: BootstrapResult | None = None
    roc_auc: float | None = None
    precision_at_k_global: dict[int, float] = field(default_factory=dict)
    recall_at_fixed_fpr: dict[float, float] = field(default_factory=dict)
    score_resolution: ScoreResolution | None = None
    auc_pr_campaign_weighted: float | None = None
    #: The same budgets as thresholds fixed on the calibration partition (S6-25).
    fixed_threshold: list[dict[str, object]] = field(default_factory=list)
    #: The budgets counted in edge-hours and user-days rather than events (S6-29).
    alert_units: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "model": self.model_name,
            "auc_pr": self.auc_pr,
            # Every campaign counts once, as in campaign recall; plain AUC-PR weights them by size.
            "auc_pr_campaign_weighted": self.auc_pr_campaign_weighted,
            "operational": {
                "budgets": self.curve.budgets,
                "event_recall": self.curve.event_recall,
                "campaign_recall": self.curve.campaign_recall,
                # What the campaign_recall row alone cannot say. The bracket is
                # how far a different tie-break could move each point, and
                # `tie_exposure` is why: a recall computed from 8 slots that
                # 2,619 equally-scored events competed for is a statement about
                # the ordering at least as much as about the model.
                "campaign_recall_tie_bracket": [
                    [low, high]
                    for low, high in zip(
                        self.curve.campaign_recall_min,
                        self.curve.campaign_recall_max,
                        strict=True,
                    )
                ],
                "tie_exposure": [exposure.__dict__ for exposure in self.curve.tie_exposure],
                # The number that does not saturate at zero: how large a daily
                # budget this model would need before it caught anything at
                # all. `None` means no budget reaches a campaign.
                "budget_for_first_detection": self.curve.budget_for_first_detection,
                "time_to_detection": [ttd.__dict__ for ttd in self.time_to_detection],
                # A SOC sets a threshold, not a daily top-k: the volume it then gets varies by day.
                "fixed_threshold": self.fixed_threshold,
                # An analyst triages a connection in an hour, or a user in a day, once.
                "alert_units": self.alert_units,
            },
            "literature_comparable": {
                "roc_auc": self.roc_auc,
                "precision_at_k_global": {str(k): v for k, v in self.precision_at_k_global.items()},
                "recall_at_fixed_fpr": {
                    f"{fpr:g}": v for fpr, v in self.recall_at_fixed_fpr.items()
                },
            },
        }
        if self.score_resolution is not None:
            # Outside both registers on purpose: it is a property of the score
            # itself, and it qualifies a metric in each of them — ROC-AUC's tie
            # credit on one side, the alert-budget tie-break on the other.
            record["score_resolution"] = self.score_resolution.__dict__
        if self.auc_pr_ci is not None:
            record["auc_pr_ci_low"] = self.auc_pr_ci.ci_low
            record["auc_pr_ci_high"] = self.auc_pr_ci.ci_high
            record["auc_pr_ci_confidence"] = self.auc_pr_ci.confidence
            record["auc_pr_ci_method"] = self.auc_pr_ci.method
            # How far each endpoint could move with another seed (95% Monte Carlo range).
            record["auc_pr_ci_monte_carlo"] = {
                "low": self.auc_pr_ci.ci_low_mc,
                "high": self.auc_pr_ci.ci_high_mc,
            }
        return record


def evaluate_model(
    stats: statistics.ModelStatistics,
    *,
    budgets: list[int],
    fpr_targets: list[float] | None = None,
    report_roc_auc: bool = True,
    warn_roc_auc: bool = True,
    auc_pr_ci: BootstrapResult | None = None,
) -> ModelEvaluation:
    """Every reported metric for one model, from its statistics on the split."""
    # ROC-AUC and recall-at-FPR are undefined without both classes; a test split with no
    # positives is a split bug, but it must not surface as a stack trace from a metric.
    both_classes_present = stats.positives.height > 0 and stats.n_benign > 0
    if report_roc_auc and warn_roc_auc:
        warnings.warn(ROC_AUC_WARNING, stacklevel=2)
    return ModelEvaluation(
        model_name=stats.model_name,
        auc_pr=statistics.average_precision(stats),
        auc_pr_ci=auc_pr_ci,
        curve=statistics.budget_curve(stats, budgets),
        time_to_detection=[TimeToDetectionSummary.from_statistics(stats, k) for k in budgets],
        roc_auc=statistics.roc_auc(stats) if report_roc_auc else None,
        precision_at_k_global={k: statistics.precision_at_k(stats, k) for k in budgets},
        recall_at_fixed_fpr=(
            {fpr: statistics.recall_at_fixed_fpr(stats, fpr) for fpr in (fpr_targets or [])}
            if both_classes_present
            else {}
        ),
        score_resolution=ScoreResolution.from_statistics(stats),
        auc_pr_campaign_weighted=(
            statistics.average_precision(stats, campaign_weighted=True)
            if both_classes_present
            else None
        ),
        fixed_threshold=(
            statistics.fixed_threshold(stats, budgets) if stats.alert_thresholds.size else []
        ),
        alert_units=statistics.unit_curves(stats, budgets),
    )


PAIRWISE_METHODS = ("paired_bootstrap",)


def pairwise_comparisons(
    bootstrap: PairedBootstrap, *, method: str = "paired_bootstrap", alpha: float = 0.05
) -> list[PairwiseComparison]:
    """Pairwise AUC-PR comparisons across the whole model family (US-128).

    The paired campaign bootstrap is the only test: it reuses the resampling pass that
    produced the per-model intervals and respects campaign-level dependence. The
    per-event permutation test it replaced assumed events exchangeable, which campaign
    structure violates, and needed every event's score in memory.
    """
    if method not in PAIRWISE_METHODS:
        raise ValueError(
            f"Unknown pairwise test method {method!r}. Supported: {', '.join(PAIRWISE_METHODS)}."
        )
    return bootstrap.comparisons("auc_pr", alpha=alpha)
