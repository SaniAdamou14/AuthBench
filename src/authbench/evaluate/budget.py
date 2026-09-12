"""Recall under a realistic per-day alert budget (US-126) — the project's central metric.

Alerts are the top-k scored events **per day**, never the top-k over the
whole test period — a SOC analyst has a daily capacity, not a period-long
one, and taking a global top-k would let a model "spend" its whole budget on
one easy day and ignore the rest (spec section 6.3 / US-126 acceptance
criteria).

**Ties are part of the metric, not an implementation detail.** A budget of 10
asks which ten events an analyst sees, and a model whose score takes few
distinct values does not answer that question on its own: measured on the demo
test split, M0b always-fail puts 2,619 events on the same score competing for
8 of day 11's slots. Something has to order them, and for most of this
module's history that something was *the row order of the frame* — which, after
`features.temporal.compute_f4`, is sorted by `(src_user, time)`. The published
floor row was therefore "the failures of the alphabetically earliest users",
which is not a property of any model.

Two things follow, and this module implements both:

- the order is now **explicit and label-free** (`ALERT_ORDER_TIE_BREAK`):
  equal scores are taken in arrival order, which is what a queue does and what
  an analyst working a shift actually does;
- the exposure is **reported** (`tie_exposure`, `campaign_recall_bracket`), so
  a reader can see how far a different tie-break could move the number instead
  of having to trust that it could not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

DEFAULT_BUDGETS: list[int] = [10, 50, 100, 500]

#: What orders two events a model scored identically.
#:
#: Arrival order, then `event_id` purely to make the order total — timestamps
#: repeat in a dataset with second resolution and millions of events a day, and
#: a metric that depends on which of two same-second events a sort happens to
#: emit first is not reproducible. Neither key looks at a label.
ALERT_ORDER_TIE_BREAK: list[str] = ["time", "event_id"]

#: How a tie is resolved when measuring what the tie could cost or buy.
#:
#: `queue` is the reported operating point. The other two are measurement
#: devices and do read `is_malicious`: they exist to bracket the number, never
#: to produce it.
TieBreak = Literal["queue", "against_model", "for_model"]

_RANK = "_rank_in_day"
_ORDER = "_alert_order"
_CAMPAIGN_EVENT_INDEX = "_campaign_event_index"


def _sort_keys(tie_break: TieBreak) -> tuple[list[str], list[bool]]:
    if tie_break == "queue":
        return ([*ALERT_ORDER_TIE_BREAK], [False, False])
    if tie_break == "against_model":
        # Benign first inside a tie group: the fewest malicious events any
        # tie-break can admit.
        return (["is_malicious", *ALERT_ORDER_TIE_BREAK], [False, False, False])
    # Malicious first, and one event per campaign before a second from any —
    # spending a slot on a campaign already covered buys no recall.
    return (
        ["is_malicious", _CAMPAIGN_EVENT_INDEX, *ALERT_ORDER_TIE_BREAK],
        [True, False, False, False],
    )


def _ranked(frame: pl.DataFrame, score_col: str, tie_break: TieBreak = "queue") -> pl.DataFrame:
    """`frame` plus each event's 1-based position in its own day's alert queue."""
    prepared = frame
    if tie_break == "for_model":
        prepared = frame.sort(["campaign_id", *ALERT_ORDER_TIE_BREAK]).with_columns(
            pl.int_range(0, pl.len()).over("campaign_id").alias(_CAMPAIGN_EVENT_INDEX)
        )

    tie_keys, tie_descending = _sort_keys(tie_break)
    ordered = prepared.sort([score_col, *tie_keys], descending=[True, *tie_descending])

    # Ranked off a unique global position rather than off the score, so the
    # rank means "place in the queue" and cannot itself be ambiguous.
    return ordered.with_row_index(_ORDER).with_columns(
        pl.col(_ORDER).rank(method="ordinal").over("day").alias(_RANK)
    )


def alerts_at_budget(
    frame: pl.DataFrame, score_col: str, k: int, *, tie_break: TieBreak = "queue"
) -> pl.DataFrame:
    """The top-`k` scored events of each day — the alert set a budget-`k` SOC
    analyst would actually see.

    Events the model scores equally are taken in arrival order; see the module
    docstring for why that is a stated choice rather than a default.
    """
    return _ranked(frame, score_col, tie_break).filter(pl.col(_RANK) <= k)


def event_recall_at_budget(
    frame: pl.DataFrame, score_col: str, k: int, *, tie_break: TieBreak = "queue"
) -> float:
    """Fraction of malicious *events* (not campaigns) captured by the daily top-k alerts."""
    total_positives = frame.filter(pl.col("is_malicious")).height
    if total_positives == 0:
        return 0.0
    alerts = alerts_at_budget(frame, score_col, k, tie_break=tie_break)
    n_true_positives = alerts.filter(pl.col("is_malicious")).height
    return n_true_positives / total_positives


def campaign_recall_at_budget(
    frame: pl.DataFrame, score_col: str, k: int, *, tie_break: TieBreak = "queue"
) -> float:
    """Fraction of campaigns with at least one event captured by the daily
    top-k alerts — the metric that reflects a defender's actual objective
    (US-106): detecting the attack, not every one of its events.
    """
    total_campaigns = frame.filter(pl.col("campaign_id").is_not_null())["campaign_id"].n_unique()
    if total_campaigns == 0:
        return 0.0
    alerts = alerts_at_budget(frame, score_col, k, tie_break=tie_break)
    detected_campaigns = alerts.filter(pl.col("campaign_id").is_not_null())[
        "campaign_id"
    ].n_unique()
    return detected_campaigns / total_campaigns


def campaign_recall_bracket(frame: pl.DataFrame, score_col: str, k: int) -> tuple[float, float]:
    """Campaign recall with every tie broken against the model, and for it.

    The low end is **exact**: putting benign events first inside every tie
    group admits the fewest malicious events any ordering can, and campaign
    detection only grows with the set of malicious events alerted.

    The high end is **achievable rather than provably maximal**: it is what one
    concrete ordering delivers — malicious first, and a campaign's first event
    before any campaign's second, so no slot is spent on a campaign already
    covered. That is optimal within a day; across days a cleverer assignment of
    slots to campaigns could in principle do better, so read this as "at least
    this much", not "at most".

    A bracket of zero width means the reported number does not depend on the
    tie-break at all, which is the common case for a model with a continuous
    score. A wide one means the number is mostly a property of the ordering.
    """
    return (
        campaign_recall_at_budget(frame, score_col, k, tie_break="against_model"),
        campaign_recall_at_budget(frame, score_col, k, tie_break="for_model"),
    )


@dataclass(frozen=True)
class TieExposure:
    """How much of a budget-`k` alert set was decided by a tie rather than a score.

    Summed over days, counting only days where the tie is actually contended —
    a tie group that fits entirely inside the remaining slots decides nothing.
    """

    budget: int
    #: Alert slots left once every strictly-higher-scored event is in.
    n_contested_slots: int = 0
    #: Events sharing the threshold score and competing for those slots.
    n_contenders: int = 0
    #: Days on which that competition happened.
    n_contested_days: int = 0

    @property
    def is_contested(self) -> bool:
        return self.n_contenders > self.n_contested_slots


def tie_exposure(frame: pl.DataFrame, score_col: str, k: int) -> TieExposure:
    """Measure the tie at the budget-`k` threshold, per day and summed.

    Published next to the recall it qualifies, because "0% at budget 500" reads
    very differently once you know 2,635 events were competing for the last 446
    of those 500 places.
    """
    if k <= 0:
        return TieExposure(budget=k)

    ranked = _ranked(frame, score_col, "queue")
    contested_slots = contenders = contested_days = 0

    for (_day,), day_frame in ranked.group_by("day", maintain_order=True):
        if day_frame.height <= k:
            # The whole day fits in the budget; nothing is left out, so nothing
            # is decided by an ordering.
            continue
        threshold = day_frame.filter(pl.col(_RANK) == k)[score_col].item()
        n_above = day_frame.filter(pl.col(score_col) > threshold).height
        n_tied = day_frame.filter(pl.col(score_col) == threshold).height
        slots = k - n_above
        if n_tied <= slots:
            continue
        contested_slots += slots
        contenders += n_tied
        contested_days += 1

    return TieExposure(
        budget=k,
        n_contested_slots=contested_slots,
        n_contenders=contenders,
        n_contested_days=contested_days,
    )


def campaign_detection_budgets(frame: pl.DataFrame, score_col: str) -> pl.DataFrame:
    """The smallest daily alert budget at which each campaign is detected.

    One row per campaign: `campaign_id`, `budget_for_detection`.

    This is the column that makes a table of zeros readable. "0% at 10, 50, 100
    and 500" says every model failed and says nothing about how far any of them
    was from succeeding — a model whose best campaign sits at rank 600 and one
    whose best sits at rank 9,000,000 print identically. A campaign is detected
    at budget `k` exactly when one of its events reaches rank `k` or better in
    its own day, so the answer per campaign is the minimum rank over its
    events: one pass, no sweep over candidate budgets, and exact.

    Ranks come from the reported `queue` ordering, so the number is the budget
    at which detection happens *at the reported operating point* — a model with
    a heavily tied score should be read together with `tie_exposure`.
    """
    ranked = _ranked(frame, score_col, "queue")
    return (
        ranked.filter(pl.col("campaign_id").is_not_null())
        .group_by("campaign_id")
        .agg(pl.col(_RANK).min().cast(pl.Int64).alias("budget_for_detection"))
        .sort("budget_for_detection", "campaign_id")
    )


def budget_for_first_detection(frame: pl.DataFrame, score_col: str) -> int | None:
    """The smallest daily alert budget at which this model detects *anything*.

    Unlike campaign recall at a fixed budget, this never saturates: it is the
    number that separates a model which missed by ten alerts from one which
    missed by six orders of magnitude. It is also always finite when there is
    anything to detect — a rank cannot exceed the size of its own day — so
    `None` means "no campaign in this split", never "never detected".
    """
    budgets = campaign_detection_budgets(frame, score_col)
    if not budgets.height:
        return None
    return int(budgets["budget_for_detection"].min())  # type: ignore[arg-type]


@dataclass
class BudgetCurve:
    model_name: str
    budgets: list[int]
    event_recall: list[float]
    campaign_recall: list[float]
    #: Campaign recall under the worst and best tie-break, per budget. Equal to
    #: `campaign_recall` wherever the score has no ties at the threshold.
    campaign_recall_min: list[float] = field(default_factory=list)
    campaign_recall_max: list[float] = field(default_factory=list)
    #: One `TieExposure` per budget, in the same order.
    tie_exposure: list[TieExposure] = field(default_factory=list)
    #: Smallest budget at which any campaign is detected. Always a real budget
    #: when the split holds a campaign at all — a rank is bounded by the size
    #: of its own day, so there is no "never"; `None` means no campaign to
    #: detect, not a failure to detect one.
    budget_for_first_detection: int | None = None


def compute_budget_curve(
    frame: pl.DataFrame, score_col: str, model_name: str, budgets: list[int] | None = None
) -> BudgetCurve:
    """Event- and campaign-recall at every budget in `budgets` — this is what
    the report's principal figure (campaign recall vs. budget, one curve per
    model) is built from (US-126).

    Also carries what the curve alone cannot say: how much of each point the
    tie-break decided, and the budget at which this model first detects
    anything.
    """
    budgets = budgets or DEFAULT_BUDGETS
    brackets = [campaign_recall_bracket(frame, score_col, k) for k in budgets]
    return BudgetCurve(
        model_name=model_name,
        budgets=budgets,
        event_recall=[event_recall_at_budget(frame, score_col, k) for k in budgets],
        campaign_recall=[campaign_recall_at_budget(frame, score_col, k) for k in budgets],
        campaign_recall_min=[low for low, _ in brackets],
        campaign_recall_max=[high for _, high in brackets],
        tie_exposure=[tie_exposure(frame, score_col, k) for k in budgets],
        budget_for_first_detection=budget_for_first_detection(frame, score_col),
    )
