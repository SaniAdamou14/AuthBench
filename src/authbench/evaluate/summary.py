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
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from authbench.evaluate.budget import BudgetCurve, compute_budget_curve
from authbench.evaluate.campaign import time_to_detection
from authbench.evaluate.metrics import auc_pr, precision_at_k, recall_at_fixed_fpr, roc_auc
from authbench.evaluate.stats_tests import (
    BootstrapResult,
    PairedBootstrap,
    PairwiseComparison,
    compare_models,
)

logger = logging.getLogger(__name__)

# The only columns evaluation needs. The bootstrap materializes a full-size
# copy of the frame on every resample, so carrying the ~40 feature columns
# through it would multiply peak memory for nothing (NFR-01).
EVAL_COLUMNS = ["event_id", "time", "day", "is_malicious", "campaign_id"]


def score_column(model_name: str) -> str:
    """Where a model's scores live on the shared evaluation frame.

    Every model's scores sit on one frame, under its own column, so a single
    campaign-stratified resampling pass can serve all of them at once —
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
    def from_frame(
        cls, frame: pl.DataFrame, score_col: str, k: int, model_name: str
    ) -> TimeToDetectionSummary:
        result = time_to_detection(frame, score_col, k, model_name)
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

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "model": self.model_name,
            "auc_pr": self.auc_pr,
            "operational": {
                "budgets": self.curve.budgets,
                "event_recall": self.curve.event_recall,
                "campaign_recall": self.curve.campaign_recall,
                "time_to_detection": [ttd.__dict__ for ttd in self.time_to_detection],
            },
            "literature_comparable": {
                "roc_auc": self.roc_auc,
                "precision_at_k_global": {str(k): v for k, v in self.precision_at_k_global.items()},
                "recall_at_fixed_fpr": {
                    f"{fpr:g}": v for fpr, v in self.recall_at_fixed_fpr.items()
                },
            },
        }
        if self.auc_pr_ci is not None:
            record["auc_pr_ci_low"] = self.auc_pr_ci.ci_low
            record["auc_pr_ci_high"] = self.auc_pr_ci.ci_high
            record["auc_pr_ci_confidence"] = self.auc_pr_ci.confidence
        return record


def evaluate_model(
    scored: pl.DataFrame,
    score_col: str,
    model_name: str,
    *,
    budgets: list[int],
    fpr_targets: list[float] | None = None,
    report_roc_auc: bool = True,
    warn_roc_auc: bool = True,
    auc_pr_ci: BootstrapResult | None = None,
) -> ModelEvaluation:
    """Every reported metric for one model, from one already-scored frame.

    `scored` must carry `is_malicious`, `campaign_id`, `day` and `time`
    alongside `score_col` — the alert-budget metrics rank *within a day*, and
    time-to-detection needs the timestamps.
    """
    y_true = scored["is_malicious"].to_numpy()
    scores = scored[score_col].to_numpy()

    # ROC-AUC and recall-at-FPR are undefined without both classes present,
    # and sklearn raises rather than returning NaN. A test split with no
    # positives is a split bug, but it must not surface as a stack trace from
    # inside a metric.
    both_classes_present = 0 < int(np.count_nonzero(y_true)) < y_true.size

    return ModelEvaluation(
        model_name=model_name,
        auc_pr=auc_pr(y_true, scores),
        auc_pr_ci=auc_pr_ci,
        curve=compute_budget_curve(scored, score_col, model_name, budgets=budgets),
        time_to_detection=[
            TimeToDetectionSummary.from_frame(scored, score_col, k, model_name) for k in budgets
        ],
        roc_auc=(
            roc_auc(y_true, scores, warn=warn_roc_auc)
            if report_roc_auc and both_classes_present
            else None
        ),
        precision_at_k_global={k: precision_at_k(y_true, scores, k) for k in budgets},
        recall_at_fixed_fpr=(
            {fpr: recall_at_fixed_fpr(y_true, scores, fpr) for fpr in (fpr_targets or [])}
            if both_classes_present
            else {}
        ),
    )


PAIRWISE_METHODS = ("paired_bootstrap", "permutation")


def pairwise_comparisons(
    scored: pl.DataFrame,
    model_names: list[str],
    bootstrap: PairedBootstrap,
    *,
    method: str = "paired_bootstrap",
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    seed: int = 42,
) -> list[PairwiseComparison]:
    """Pairwise AUC-PR comparisons across the whole model family (US-128).

    `paired_bootstrap` reuses the resampling pass that already produced the
    per-model CIs, so the intervals and the comparisons come from one
    sampling distribution, and it respects campaign-level dependence.

    `permutation` is the exact per-event paired test. It assumes the two
    models' scores are exchangeable event by event — which is precisely the
    independence assumption campaign structure violates — and it costs
    `n_permutations x n_pairs` metric evaluations. Available for small
    splits; not the default.
    """
    if method == "paired_bootstrap":
        return bootstrap.comparisons("auc_pr", alpha=alpha)

    if method == "permutation":
        n_pairs = len(model_names) * (len(model_names) - 1) // 2
        logger.warning(
            "pairwise_test.method=permutation: %d pairs x %d permutations = %d AUC-PR "
            "evaluations over %d events, and the per-event exchangeability it assumes "
            "ignores campaign structure. paired_bootstrap is the default for both reasons.",
            n_pairs,
            n_permutations,
            2 * n_pairs * n_permutations,
            scored.height,
        )
        return compare_models(
            model_names,
            scored["is_malicious"].to_numpy(),
            {name: scored[score_column(name)].to_numpy() for name in model_names},
            auc_pr,
            "auc_pr",
            n_permutations=n_permutations,
            alpha=alpha,
            seed=seed,
        )

    raise ValueError(
        f"Unknown pairwise test method {method!r}. Supported: {', '.join(PAIRWISE_METHODS)}."
    )
