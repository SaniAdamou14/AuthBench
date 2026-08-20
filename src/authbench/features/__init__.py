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

__all__ = ["MODEL_FEATURE_COLUMNS"]
