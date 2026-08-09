"""M1 — the heuristic rule engine (US-118, spec section 5.1).

This is deliberately not a strawman: it is the real competitor to the deep
models, and the project loses its point if M1 is under-optimized. Each rule
produces a raw signal, is rank-normalized independently (so no rule dominates
merely because of its numeric scale), and the seven rank-normalized scores
are combined by a weighted sum whose weights are calibrated by random search
on the validation period only.

Every rule carries a MITRE ATT&CK mapping as metadata — the bridge between
the ML and security framings of the project (spec section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from authbench.features.causal import causal_first_occurrence, sort_for_causal
from authbench.models.base import BaseAnomalyScorer


class NoPositivesInValidationError(ValueError):
    """Raised when M1's weight calibration is asked to optimize AUC-PR against
    a validation period containing no malicious events — a degenerate objective
    that would otherwise return arbitrary weights while looking successful.
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
]


def _rank_normalize(expr: pl.Expr) -> pl.Expr:
    """Percentile rank in [0, 1] — the common scale every rule score is combined on."""
    return expr.rank(method="average") / pl.len()


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


def _rule_r6(new_auth_type: pl.LazyFrame) -> pl.LazyFrame:
    return new_auth_type


def _rule_r7(events: pl.LazyFrame, chain_window_seconds: int) -> pl.LazyFrame:
    by_user = sort_for_causal(events, "src_user", "time")
    with_prev = by_user.with_columns(
        [
            pl.col("dst_computer").shift(1).over("src_user").alias("_prev_dst"),
            pl.col("time").shift(1).over("src_user").alias("_prev_time"),
        ]
    )
    return with_prev.select(
        [
            "event_id",
            (
                (pl.col("_prev_dst") == pl.col("src_computer"))
                & ((pl.col("time") - pl.col("_prev_time")) <= chain_window_seconds)
            )
            .fill_null(False)
            .cast(pl.Float64)
            .alias("R7_lateral_chain"),
        ]
    )


class RulesScorer(BaseAnomalyScorer):
    """M1. Requires the F1-F4 feature columns to already be present on the
    frames passed to `fit`/`calibrate_weights`/`score`.
    """

    name = "M1_rules"
    requires_labels = False

    def __init__(
        self, chain_window_seconds: int = 1800, percentile_threshold: float = 99.9
    ) -> None:
        self.chain_window_seconds = chain_window_seconds
        self.percentile_threshold = percentile_threshold
        self._user_host_thresholds: dict[str, float] = {}
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

    def _raw_rule_scores(self, features: pl.LazyFrame) -> pl.LazyFrame:
        new_auth_type = causal_first_occurrence(
            features, group_cols=["src_user", "auth_type"], time_col="time", id_col="event_id"
        ).select(["event_id", pl.col("is_new").cast(pl.Float64).alias("R6_new_auth_type")])

        chain = _rule_r7(features, self.chain_window_seconds)

        base = features.select(
            [
                "event_id",
                _rule_r1(features).alias("R1_new_pair"),
                _rule_r2(features, self._user_host_thresholds).alias("R2_host_burst"),
                _rule_r3(features).alias("R3_fail_then_success"),
                _rule_r4(features).alias("R4_offhours"),
                _rule_r5(features).alias("R5_machine_account_anomaly"),
            ]
        )

        # `maintain_order="left"` is load-bearing, not cosmetic. R6 and R7 are
        # computed on frames re-sorted by (group key, time), and every caller
        # attaches `score()`'s output back onto its input frame *by position*
        # (`frame.with_columns(pl.Series("_score", scores))`). A join that
        # reordered rows would silently attribute each event's score to a
        # different event — the results table would still look entirely
        # plausible. Polars does not guarantee row order on a join unless it
        # is asked to, so it is asked to.
        return base.join(new_auth_type, on="event_id", how="left", maintain_order="left").join(
            chain, on="event_id", how="left", maintain_order="left"
        )

    def rule_scores(self, features: pl.LazyFrame) -> pl.DataFrame:
        """Rank-normalized per-rule scores — published as its own table
        (US-118: "avant agrégation") because individual rule performance is
        what a security jury actually cares about.
        """
        raw = self._raw_rule_scores(features)
        rule_ids = [r.id for r in RULES]
        return raw.select(
            ["event_id", *[_rank_normalize(pl.col(rid)).alias(rid) for rid in rule_ids]]
        ).collect()

    def calibrate_weights(
        self,
        val_features: pl.LazyFrame,
        val_labels: pl.Series,
        *,
        n_trials: int = 200,
        seed: int = 42,
    ) -> None:
        """Random search over the weight simplex, maximizing AUC-PR on the
        validation period only (US-118). The seed is stored so the search is
        reproducible bit-for-bit (NFR-02).

        Raises `NoPositivesInValidationError` if the validation period holds no
        malicious events: `average_precision_score` returns 0.0 for every
        candidate in that case, so the search would keep its first arbitrary
        draw and report a "calibrated" M1 that was never calibrated at all.
        A validation window with no positives is a split/data-generation bug,
        and it must not be able to hide behind a plausible-looking result.
        """
        from sklearn.metrics import average_precision_score

        rule_ids = [r.id for r in RULES]
        scores_df = self.rule_scores(val_features)
        matrix = scores_df.select(rule_ids).to_numpy()
        y = val_labels.to_numpy()

        n_positives = int(np.count_nonzero(y))
        if n_positives == 0:
            raise NoPositivesInValidationError(
                f"M1 weight calibration needs malicious events in the validation period, "
                f"but all {y.size} validation labels are negative. AUC-PR is identically 0 "
                "for every weight vector, so the random search would silently return its "
                "first arbitrary draw. Check the val_days window of your split config "
                "against the red-team event timestamps."
            )

        rng = np.random.default_rng(seed)
        best_weights, best_ap = None, -1.0
        for _ in range(n_trials):
            raw_weights = rng.dirichlet(np.ones(len(rule_ids)))
            combined = matrix @ raw_weights
            ap = average_precision_score(y, combined)
            if ap > best_ap:
                best_ap, best_weights = ap, raw_weights

        assert best_weights is not None
        self.weights = dict(zip(rule_ids, best_weights.tolist(), strict=True))

    def score(self, data: pl.LazyFrame) -> pl.Series:
        scores_df = self.rule_scores(data)
        rule_ids = [r.id for r in RULES]
        combined = np.zeros(scores_df.height)
        for rid in rule_ids:
            combined += scores_df[rid].to_numpy() * self.weights.get(rid, 0.0)
        return pl.Series(self.name, combined)
