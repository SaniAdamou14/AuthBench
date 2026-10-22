"""M1 — the heuristic rule engine (US-118, spec section 5.1).

This is deliberately not a strawman: it is the real competitor to the deep
models, and the project loses its point if M1 is under-optimized. Each rule
produces a raw signal, is mapped to [0, 1] by its empirical CDF fitted on the
last training day (so no rule dominates by its numeric scale and no score
depends on another scored event), and the eight scores are combined by a
weighted sum. M1's weights are chosen on the validation period by
`pipeline.tuning`, from the same number of candidates as every other tuned
model's hyperparameters; M1u keeps uniform weights and reads no label.

Every rule carries a MITRE ATT&CK mapping as metadata — the bridge between
the ML and security framings of the project (spec section 5.1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import polars as pl

from authbench.models.base import BaseAnomalyScorer


class NoPositivesInValidationError(ValueError):
    """Raised when tuning is asked to optimize AUC-PR against a validation
    period containing no malicious events — a degenerate objective that would
    otherwise return the first candidate while looking successful.
    """


@dataclass(frozen=True)
class RuleSpec:
    id: str
    mitre: str
    description: str


RULES: list[RuleSpec] = [
    RuleSpec("R1_new_pair", "T1078", "First-ever (user, destination) pair"),
    RuleSpec("R2_host_burst", "T1078", "Distinct hosts in 1h beyond user's 99.9th percentile"),
    RuleSpec("R3_fail_then_success", "T1110", "Failure burst on a destination followed by success"),
    RuleSpec("R4_offhours", "T1078", "Authentication outside the user's usual hours"),
    RuleSpec("R5_machine_account_anomaly", "T1550", "Machine account behaving atypically"),
    RuleSpec("R6_new_auth_type", "T1550", "Auth type never used before by this user"),
    RuleSpec("R7_lateral_chain", "T1021", "A→B then B→C chain, same user, short window"),
    RuleSpec("R8_new_source", "T1021", "First use of a source computer by this user"),
]


def _ecdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Mid-rank empirical CDF of `values` against a sorted training reference, in [0, 1]."""
    below = np.searchsorted(reference, values, side="left")
    at_or_below = np.searchsorted(reference, values, side="right")
    return (below + at_or_below) / (2.0 * reference.size)


def _rule_r1(features: pl.LazyFrame) -> pl.Expr:
    return pl.col("pair_is_new").cast(pl.Float64)


def _rule_r2(features: pl.LazyFrame, thresholds: dict[str, float]) -> pl.Expr:
    default = float(np.mean(list(thresholds.values()))) if thresholds else 0.0
    threshold_map = (
        pl.col("src_user")
        .cast(pl.Utf8)
        .replace_strict(thresholds, default=default, return_dtype=pl.Float64)
    )
    return (pl.col("src_user_1h_n_distinct_dst") - threshold_map).clip(lower_bound=0)


def _rule_r3(features: pl.LazyFrame) -> pl.Expr:
    return pl.col("dst_computer_1h_n_failures").cast(pl.Float64) * pl.col("is_success").cast(
        pl.Float64
    )


def _rule_r4(features: pl.LazyFrame) -> pl.Expr:
    return pl.col("hour_deviation_from_profile")


def _rule_r5(features: pl.LazyFrame) -> pl.Expr:
    return pl.col("src_user_is_machine").cast(pl.Float64) * pl.col("pair_global_rarity")


class RulesScorer(BaseAnomalyScorer):
    """M1. Requires the F1-F4 feature columns to already be present on the
    frames passed to `fit`/`score`.
    """

    name = "M1_rules"
    # Unsupervised fit, but the weights are chosen on validation labels (`pipeline.tuning`).
    requires_labels = True
    # R6 and R7 come precomputed from F3/F4 and the rule scale is fitted on training data,
    # so an event's score depends on that event's row alone.
    scores_row_locally = True

    def __init__(
        self,
        chain_window_seconds: int = 1800,
        percentile_threshold: float = 99.9,
        normalization_days: int = 1,
        normalization_sample_size: int = 2_000_000,
        seed: int = 42,
    ) -> None:
        self.chain_window_seconds = chain_window_seconds
        self.percentile_threshold = percentile_threshold
        self.normalization_days = normalization_days
        self.normalization_sample_size = normalization_sample_size
        self.seed = seed
        self._user_host_thresholds: dict[str, float] = {}
        self._reference: dict[str, np.ndarray] = {}
        self.weights: dict[str, float] = {r.id: 1.0 / len(RULES) for r in RULES}

    def fit(self, train: pl.LazyFrame) -> None:
        """Learns each user's R2 threshold (99.9th percentile of their own
        1h distinct-host-count history) from the training period only.
        """
        per_user = (
            train.group_by("src_user")
            .agg(pl.col("src_user_1h_n_distinct_dst").quantile(self.percentile_threshold / 100.0))
            .collect()
        )
        self._user_host_thresholds = dict(
            zip(
                per_user["src_user"].cast(pl.Utf8).to_list(),
                per_user["src_user_1h_n_distinct_dst"].to_list(),
                strict=True,
            )
        )
        # Rule scales come from the last training days, so no scored event depends on another.
        last_day = train.select(pl.col("day").max()).collect().item()
        window = train.filter(pl.col("day") > last_day - self.normalization_days)
        raw = self._raw_rule_scores(window).collect()
        if raw.height > self.normalization_sample_size:
            raw = raw.sample(n=self.normalization_sample_size, seed=self.seed)
        self._reference = {r.id: np.sort(raw[r.id].to_numpy()) for r in RULES}

    def _raw_rule_scores(self, features: pl.LazyFrame) -> pl.LazyFrame:
        chain = pl.col("prev_dst_is_src") & (
            pl.col("hours_since_prev_event_same_user") * 3600 <= self.chain_window_seconds
        )
        return features.select(
            "event_id",
            _rule_r1(features).alias("R1_new_pair"),
            _rule_r2(features, self._user_host_thresholds).alias("R2_host_burst"),
            _rule_r3(features).alias("R3_fail_then_success"),
            _rule_r4(features).alias("R4_offhours"),
            _rule_r5(features).alias("R5_machine_account_anomaly"),
            pl.col("user_auth_is_new").cast(pl.Float64).alias("R6_new_auth_type"),
            chain.cast(pl.Float64).alias("R7_lateral_chain"),
            pl.col("user_src_is_new").cast(pl.Float64).alias("R8_new_source"),
        )

    def rule_scores(self, features: pl.LazyFrame) -> pl.DataFrame:
        """Per-rule scores on the training-fitted empirical CDF of each rule (US-118)."""
        if not self._reference:
            raise RuntimeError("RulesScorer.fit() must run before rule_scores().")
        raw = self._raw_rule_scores(features).collect()
        return raw.select("event_id").with_columns(
            pl.Series(r.id, _ecdf(raw[r.id].to_numpy(), self._reference[r.id])) for r in RULES
        )

    def with_weights(self, weights: np.ndarray) -> RulesScorer:
        """This fitted engine with other rule weights, in `RULES` order; the fit is shared."""
        variant = copy.copy(self)
        variant.weights = dict(zip([r.id for r in RULES], weights.tolist(), strict=True))
        return variant

    def score(self, data: pl.LazyFrame) -> pl.Series:
        scores_df = self.rule_scores(data)
        rule_ids = [r.id for r in RULES]
        combined = np.zeros(scores_df.height)
        for rid in rule_ids:
            combined += scores_df[rid].to_numpy() * self.weights.get(rid, 0.0)
        return pl.Series(self.name, combined)


class UniformRulesScorer(RulesScorer):
    """M1u: the same eight rules with uniform weights, never tuned and reading no label.

    The reference that separates what the rules detect from what choosing their weights on
    validation labels adds: M1 is the only model the published runs tuned on labels.
    """

    name = "M1u_rules_uniform"
    requires_labels = False


def rule_discrimination(model: RulesScorer, features: pl.DataFrame) -> pl.DataFrame:
    """Per-rule mean score on benign vs malicious events, next to the weight M1
    actually gave that rule.

    US-118 asks for individual rule performance *before* aggregation, because
    that is what a security audience reads first. It is also the only honest
    way to read M1's headline number: a rule separating the classes by orders
    of magnitude is either a good rule or a rule that was handed the answer,
    and the aggregate score cannot tell those apart.

    On the committed demo run it says the latter — R7 (lateral chain) separates
    113-fold and carries 0.52 of the weight, because the sample generator emits
    campaigns as A→B→C chains and R7 tests for A→B→C chains: the same predicate
    on both sides of the experiment. See `reports/demo/RUN.md`.
    """
    raw = model._raw_rule_scores(features.lazy()).collect()
    labelled = raw.join(features.select(["event_id", "is_malicious"]), on="event_id")
    benign_rows = labelled.filter(~pl.col("is_malicious"))
    malicious_rows = labelled.filter(pl.col("is_malicious"))

    def _mean(frame: pl.DataFrame, column: str) -> float:
        if not frame.height:
            return 0.0
        # Aggregated through a `select`, not `Series.mean()`: the latter is
        # typed as a union covering every dtype Polars can hold a mean of,
        # which no cast on a Float64 column can narrow.
        mean = frame.select(pl.col(column).cast(pl.Float64).mean()).item()
        return float(mean) if mean is not None else 0.0

    rows = [
        {
            "rule": rule.id,
            "mitre": rule.mitre,
            "description": rule.description,
            "mean_score_benign": _mean(benign_rows, rule.id),
            "mean_score_malicious": _mean(malicious_rows, rule.id),
            # A rule that never fires on benign events has no finite ratio;
            # an explicit null beats an infinity that renders as an error in
            # every spreadsheet this table will be opened in.
            "separation_ratio": (
                _mean(malicious_rows, rule.id) / _mean(benign_rows, rule.id)
                if _mean(benign_rows, rule.id)
                else None
            ),
            "calibrated_weight": model.weights.get(rule.id, 0.0),
        }
        for rule in RULES
    ]
    # Rounded so the committed table does not depend on the platform's last floating-point digit.
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col(pl.Float64).round(12))
        .sort("calibrated_weight", descending=True)
    )
