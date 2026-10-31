# Limitations and threats to validity

Stated here deliberately, ahead of any reviewer.

1. **Temporal leakage** is the dominant pitfall in this literature. Mitigated by
   `authbench.split.temporal` (mechanical `LeakageError` guard) and
   `tests/unit/test_no_test_leakage.py` (architecture test forbidding `models/`/`features/` from
   importing the test-partition loader) — but still the first thing a reviewer should re-check.
2. **A single real dataset.** LANL describes one enterprise network, in 2015, on Windows/Active
   Directory. Nothing here establishes transferability to a modern cloud/hybrid environment.
3. **Dataset age.** Eleven years old at time of writing. Authentication patterns have shifted
   (widespread MFA, token-based auth). The *protocol* remains valid; absolute performance numbers
   are not transferable to present-day environments.
4. **Incomplete labels.** Red-team events are a partial ground truth — an event not flagged as
   red-team may still be an undetected real compromise from 2015. This means a "false positive"
   in this benchmark may in fact be an unlabeled true positive; precision figures should be read
   with that caveat.
5. **No second dataset is evaluated.** The harness downloads and configures CERT r4.2, but no
   result on it is reported. Being synthetic, CERT could test whether the protocol generalises;
   it could not support any claim about real insider behavior.
6. **Campaign-grouping sensitivity.** Campaign-level recall depends on the grouping gap
   threshold (`campaign_gap_hours`). The sensitivity of campaign count to this threshold is a
   required, non-decorative analysis. Since 22 October 2026 every run writes
   `campaign_gap_sensitivity.json`: campaigns regrouped per account at 1, 6, 24 and 72 hours
   from the positives ledger, with the campaign count and each model's recall at every budget,
   first detection, median budget and log-area. Regrouping sees only the split's positives, so a
   campaign that continues outside the split can divide differently than over the whole
   red-team file. On the demo the four test campaigns are the same at every gap.
7. **Negative subsampling**, if used under hardware constraints, must be described precisely and
   its effect on metric variance measured, not assumed negligible.
8. **No adaptive adversary.** LANL's red team was not attempting to evade these specific models.
   Reported numbers are an optimistic upper bound against an adversary who knows they're being
   watched by exactly this detection stack.

9. **One intrusion, not many campaigns.** In the wider-window run, 275 of the 282 test
   red-team events and 50 of the 52 test campaigns start from one source host, C17693
   (84.0% of training and 98.1% of validation positives too; `split_summary.json`). The
   labelled LANL activity is a single simulated intrusion (Larroche, arXiv 2607.29390).
   Campaigns are the bootstrap's resampling unit, so every interval describes variability
   across the accounts and phases of one operation, not across intrusions. On the whole
   red-team file, 683 of the 715 distinct events (95.5%) start from C17693 and only four
   source computers appear (`reports/runs/20261022T102638Z/diagnostics/redteam_concentration.json`,
   23 October 2026).
10. **Unequal access to labels.** In the published runs M1's weights were calibrated on
    validation labels and no other model read them. Since 22 October 2026 every tunable model
    gets the same budget of candidates on validation and M1u reports the rules without labels;
    the autoencoders are still not tuned. See `methodology.md`, "Labels".
11. **Transductive scores.** ECOD (distributions recomputed from the scored matrix) depends
    on the test distribution. So did M1 in the published LANL runs (rank normalisation over the
    three test days); its normalisation is fitted on training data since 22 October 2026.
    Since the same day ECOD is fitted on training data too, and every model is row-local to
    the last bit (`methodology.md`); ECOD was excluded from both published LANL runs. The full
    run of 22 October 2026 uses the fitted versions and evaluates ECOD.
12. **Novelty is relative to the window read.** The F3 `*_is_new` and `*_rarity` features
    count earlier events of the partition plus its warm-up, not the whole history. In the
    published LANL runs the rarity denominator grew with every event, so within a day pair
    rarity ranked equally new events latest first; since 22 October 2026 it is anchored to the
    start of the day, equally new events of a day tie, and the tie bracket shows the effect.
    `split.history: cumulative` (22 October 2026) compares every event with all earlier
    events from day 0; the published runs used `window`.
13. **The first-detection interval is one-sided.** `budget_summary.json` bootstraps accounts
    and takes the percentile interval of each summary. For `first_detection`, a minimum over
    campaigns, a resample can only keep or lose the best campaign, never beat it, so `ci_low`
    equals the point value by construction (22 October 2026 LANL run: all 13 models in both
    regimes of `temporal` and in the single regime of `harmonised`). Read it as an upper bound
    on how many alerts a day the first detection could need, not as a two-sided interval; the
    percentile bootstrap of an extreme is not
    consistent. The quantile budgets and the log-area are not affected in this way.
14. **Alert units carry no tie bracket.** Edge-hours and user-days take their highest event's
    score, and equal units are ordered by their first event's time. For coarse scores (M2a,
    M2c, M0b) the unit results can reflect that order. On the full run, user-day recall
    changes the picture (Euler 51.2% and source rarity 50.6% of the harmonised campaigns at
    500 user-days a day, against 0.6% and 0.0% at 500 events), and the random floor itself
    reaches 4.3%, because a user-day budget covers more of a day. That change is not
    detection: ranking each day's users by the number of distinct computers they touched,
    with no detector and no label, reaches 56.8% (`scripts/user_day_activity.py`, review
    round 3), and a user's maximum score rises with its number of events, so a uniform
    ranking of user-days is the wrong null; random event scores aggregated by their maximum
    reach 6.0%.
15. **Fitting variance is measured for the seeded classical models only.** M1, PCA,
    Isolation Forest, ECOD and HBOS are refitted with five seeds (`fit_variance.json`); on the
    harmonised protocol PCA's first detection ranges from 13 to 243 alerts a day. The
    autoencoders and Euler are fitted once, and may move as much.
16. **Thresholds fitted on a short window do not transfer.** With the harmonised protocol's
    41 training hours, a threshold calibrated for 10 alerts a day raises 149 a day on average
    on the test for PCA (8,222 on its worst day); the daily top-k hides that by construction.
17. **Euler is a partial reproduction.** On the authors' protocol the reimplementation stays
    about 3 points of AUC and half the AP below the authors' own code on identical test edges;
    their code contains training choices the paper does not state, which are not ported.
    (Addition, 23 October 2026, review round 2: the reimplementation also has two unintended
    deviations, a degree-scaled bias in the second GCN layer and a self-loop added even where
    one exists; and what was not reproduced is the authors' released code and preprocessing
    in its November 2023 environment, whose node and event counts differ from the paper's
    table V.)
18. **The harmonised first test day is partial.** Test day 1 holds hours 41 to 47 only
    (1,653,645 events against 4.6 to 8.3 million on the other days), so a daily budget is a
    larger share of it. The autoencoder's and Euler's first detections fall on that day; without
    it they move from 59 to 528 and from 203 to 397 alerts a day (report, appendix D).
19. **Design choices made after seeing results on the same operation.** The source-side
    features (M2c, R8), the edge-hour and user-day units and the day-anchored rarity were added
    after the September run's results on the temporal test days were reviewed. Those days lie
    inside the harmonised test window. Results that depend on these choices, above all the
    user-day results of source rarity, are not confirmatory.

**Correction to item 13, 23 October 2026 (review round 2).** The percentile bootstrap of a
minimum has no coverage interpretation: its upper end is, by construction, about the fourth
smallest campaign budget of the resampled accounts. The report no longer presents it as an
interval; it gives the three smallest campaign budgets, the exact null and the seed range.

## Metrics deliberately not used as primary

- **Accuracy** — a model that always predicts "benign" scores ≈99.99993%.
- **ROC-AUC as a ranking criterion** — stays high even for operationally useless models at
  LANL's ~10⁻⁷ positive rate. Reported in the annex only, always with
  `authbench.evaluate.metrics.ROC_AUC_WARNING`.

## One day of history per partition (LANL run, August 2026)

The published LANL result splits one day into train, one into validation and one
into test, with no warm-up. F2's 24-hour windows and F3's cumulative pair
statistics therefore see at most one day of past.

This makes a campaign recall of 0% ambiguous between two readings that the run
cannot separate: the models genuinely detect nothing at an operational budget,
or one day is not enough history to establish what counts as new.

The effect is measured rather than asserted. On the demo sample, adding two
warm-up days halves `pair_is_new` (2,623 -> 1,287) over an identical set of
written events: without history, half the pairs called "never seen before" were
only unseen because there was no past to have seen them in.

### Cold start is encoded as "normal", and that compounds it

An event with no prior history for its user does not get a neutral feature
value — it gets the most normal one available. `compute_f4` fills
`hour_deviation_from_profile` with `0.0` for a user's first event in a
partition, and `0.0` is the **minimum** of that column's range; the F2 failure
ratios are likewise 0 when there are no prior events to have failed.

Measured on the demo test split rather than asserted:

| | |
|---|---|
| Events with no prior 1h history | 3,299 of 8,049 (**41%**) |
| Events that are a user's first in the split | 350 |
| Of those, carrying `hour_deviation_from_profile == 0.0` | **350 — all of them, and no other event** |
| Median deviation elsewhere · 99th percentile | 0.426 · 3.046 |
| Malicious test events in that set | **2 of 11** |

So the sentinel and the cold-start set coincide exactly, and they are scored as
maximally typical on that axis. This is not independent of the one-day-of-history
limitation above — it is the same shortage seen from the feature side, and the
shorter each partition, the larger the affected share. It biases in the
direction that produces the reported zeros, which is the direction that
flatters the conclusion, so it is stated here rather than left to be found.

It is **not** fixed, and the reason is a trade rather than an oversight:
changing the sentinel changes the design matrix, which moves every
vector-space model's scores and would desynchronise
[`../reports/lanl/`](../reports/lanl/RUN.md) from the code that claims to
produce it — and that snapshot cannot be re-run cheaply. The honest options are
a cold-start indicator column or a sentinel outside the column's range, both of
which are changes to the protocol and belong with a re-run, not before one.

**Update, 21 October 2026.** The wider-window run lifts the one-day limitation:
two warm-up days and multi-day partitions. Its zeros stand (no campaign within 500
alerts a day, under any tie order), and ROC-AUC still does not track the operational
ordering (Spearman -0.10 against the budget for first detection over five models). The
earlier wording, "anti-correlated", overstated what two extreme models establish; the
data support "decoupled". The cold-start sentinel above is unchanged in that run.

See [`../reports/lanl/RUN.md`](../reports/lanl/RUN.md) and
[`scaling.md`](scaling.md).
