"""Feature families F1–F6, and the one list of columns the models actually see.

`MODEL_FEATURE_COLUMNS` lives here rather than next to any single caller
because it is a *protocol* decision, not a plumbing detail: it is the design
matrix every vector-space model (M2b PCA, M3a Isolation Forest, M3b ECOD /
HBOS, M4 autoencoders) is fitted on. Two callers keeping two copies of it —
which is how this started — means `authbench demo` and the DVC `train_eval`
stage silently benchmark *different models under the same names*, and the
comparison the whole project exists to make is no longer between the models
it claims to compare.
"""

from __future__ import annotations

# F1 — event-local, no history needed. The three `_freq` columns are the
# training-split-fitted frequency encodings (`features.event`); leaving them
# out discards the only F1 features that are not raw booleans.
_F1_COLUMNS: list[str] = [
    "is_success",
    "auth_type_is_null",
    "logon_type_is_null",
    "src_user_is_machine",
    "src_dst_user_same",
    "src_dst_computer_same",
    "domain_crossing",
    "auth_type_freq",
    "logon_type_freq",
    "auth_orientation_freq",
]

# F2 — per-entity history. A deliberate subset of the full F2 output: the
# source user's 1h and 24h volume and failure ratio. The remaining F2 columns
# (7-day windows, per-machine entities, entropy/diversity) are highly
# collinear with these and multiply the matrix width without adding rank, so
# they stay available on the feature store and out of the design matrix.
_F2_COLUMNS: list[str] = [
    "src_user_1h_n_events",
    "src_user_1h_failure_ratio",
    "src_user_1d_n_events",
    "src_user_1d_failure_ratio",
]

# F3 — pair novelty and rarity. `days_since_pair_last_seen` is excluded on
# purpose: its documented sentinel for a first-ever pair is +inf (US-111),
# which is not a number any of these estimators can standardize, and
# `pair_is_new` already carries exactly that information as a boolean.
_F3_COLUMNS: list[str] = [
    "pair_is_new",
    "pair_global_rarity",
    "user_new_host_count_24h",
    "host_new_user_count_24h",
]

# F4 — temporal. `hours_since_prev_event_same_user` is excluded for the same
# +inf reason as above.
_F4_COLUMNS: list[str] = [
    "hour_sin",
    "hour_cos",
    "hour_deviation_from_profile",
]

MODEL_FEATURE_COLUMNS: list[str] = [
    *_F1_COLUMNS,
    *_F2_COLUMNS,
    *_F3_COLUMNS,
    *_F4_COLUMNS,
]

# Identity, time and label columns every stage after `split` needs: the
# evaluation frame (`evaluate.summary.EVAL_COLUMNS`) plus the join key.
_IDENTITY_COLUMNS: list[str] = [
    "event_id",
    "time",
    "day",
    "is_malicious",
    "campaign_id",
]

# Columns the non-matrix models read directly off the feature store.
#
# M1 recomputes two of its seven rules (R6 new auth type, R7 lateral chain)
# from raw event fields rather than from precomputed features, so those fields
# have to survive into the store even though no model treats them as features.
_MODEL_INPUT_COLUMNS: list[str] = [
    "src_user",  # M1 R2 threshold lookup, R6 grouping, R7 partitioning
    "src_computer",  # M1 R7: does this event start where the last one ended?
    "dst_computer",  # M1 R7
    "auth_type",  # M1 R6
    "success",  # M0b always-fail
    "src_user_1h_n_distinct_dst",  # M1 R2
    "dst_computer_1h_n_failures",  # M1 R3
]

#: What `pipeline.build_features` actually writes to disk.
#:
#: The feature pipeline produces about ninety columns; this is the ~thirty that
#: something downstream reads. Persisting the rest cost roughly three times the
#: disk for data no stage opens — 138 GB against 46 GB at full LANL scale, on a
#: laptop where that difference decides whether the run happens at all. The
#: dropped columns are not lost, only unwritten: rerunning the stage with a
#: different projection regenerates them, and the feature-store version hash
#: already captures the configuration that produced any given store.
FEATURE_STORE_COLUMNS: list[str] = list(
    dict.fromkeys([*_IDENTITY_COLUMNS, *MODEL_FEATURE_COLUMNS, *_MODEL_INPUT_COLUMNS])
)

__all__ = ["FEATURE_STORE_COLUMNS", "MODEL_FEATURE_COLUMNS"]
