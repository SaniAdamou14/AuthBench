# Dataset notes

## LANL quirks handled explicitly (spec section 2.1)

| Quirk | Where it's handled |
|---|---|
| `auth_type` null ≈55%, `logon_type` null ≈14% | `parse.clean.clean_auth` turns the `?` sentinel into a dedicated `*_is_null` boolean, never imputes |
| Time starts at epoch 1, no timezone, 1s resolution | `parse.schema.SECONDS_PER_DAY`; day-of-week/night window calibrated empirically, not assumed (`features.temporal.calibrate_night_window`) |
| Failures only exist for users who succeeded somewhere | Documented limitation — no conclusion is drawn about brute force against nonexistent accounts |
| Machine accounts (`$` suffix) are most of the volume | `parse.clean.is_machine_account`, flagged via `src_user_is_machine` / `dst_user_is_machine`, never dropped or coerced |
| Label join must be on the full quadruplet | `label.redteam_join.label_auth_events` joins on `(time, user, domain, src_computer, dst_computer)`, never on time alone |

## Positive rate sanity check (US-105)

`conf/dataset/lanl.yaml`'s `positive_rate_bounds: [5.0e-7, 1.0e-6]` is checked against the actual
labeling output; the pipeline aborts if the observed rate falls outside this range, since a
looser join silently inflates the positive count.

## Campaign grouping (US-106, spec section 4.2)

The 749 raw red-team rows (**715** after deduplication — 34 exact duplicates, measured on the
file rather than taken from the widely-repeated secondary figure of 737) are not 715 independent
attacks — they are a handful of lateral-movement campaigns. `label.redteam_join.group_into_campaigns` groups by
`user@domain` with a configurable gap threshold (`campaign_gap_hours`, default 24h). There is no
canonical grouping in the literature; the sensitivity of the resulting campaign count to this
threshold must be reported, not treated as a fixed constant (see `docs/limitations.md`).

## CERT r4.2 malicious-user count discrepancy

Published figures for r4.2's malicious user count disagree across papers (70/1000 vs. 30 × 3
scenarios). Whichever labeling procedure is used here should be documented precisely and the
resulting count reconciled against both published figures — a small but genuine reproducibility
contribution (spec section 2.2).
