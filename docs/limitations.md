# Limitations and threats to validity

Stated here deliberately, ahead of any reviewer — see spec section 11 for the full discussion
this summarizes.

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
5. **CERT's synthetic nature.** Useful to test generalization, insufficient on its own to
   support any claim about real insider behavior.
6. **Campaign-grouping sensitivity.** Campaign-level recall depends on the grouping gap
   threshold (`campaign_gap_hours`). The sensitivity of campaign count to this threshold is a
   required, non-decorative analysis (US-106).
7. **Negative subsampling**, if used under hardware constraints, must be described precisely and
   its effect on metric variance measured, not assumed negligible.
8. **No adaptive adversary.** LANL's red team was not attempting to evade these specific models.
   Reported numbers are an optimistic upper bound against an adversary who knows they're being
   watched by exactly this detection stack.

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

What the limitation does **not** reach is the anti-correlation between ROC-AUC
and campaign recall — M2a at 0.942 detecting nothing, M1 at 0.547 being the only
model that detects anything. That is a property of the operating point and the
positive rate, and no amount of history changes it.

See [`../reports/lanl/RUN.md`](../reports/lanl/RUN.md) and
[`scaling.md`](scaling.md).
