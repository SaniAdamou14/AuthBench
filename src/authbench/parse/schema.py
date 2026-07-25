"""Single source of truth for the LANL schema, raw and typed.

Every other module imports the column lists and dtypes from here rather than
re-declaring them — a schema drift between `ingest`, `features` and `models`
is exactly the kind of silent bug that invalidates a benchmark.
"""

from __future__ import annotations

import polars as pl

# --- Raw auth.txt: 9 comma-separated fields, no header. ------------------------------

RAW_AUTH_COLUMNS: list[str] = [
    "time",
    "src_user_at_domain",
    "dst_user_at_domain",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success_failure",
]

RAW_AUTH_SCHEMA: dict[str, type[pl.DataType]] = {name: pl.Utf8 for name in RAW_AUTH_COLUMNS}
RAW_AUTH_SCHEMA["time"] = pl.Int64  # cast to Int32 after we confirm the 58-day range

# --- Raw redteam.txt: 4 comma-separated fields, no header. --------------------------

RAW_REDTEAM_COLUMNS: list[str] = [
    "time",
    "user_at_domain",
    "src_computer",
    "dst_computer",
]

RAW_REDTEAM_SCHEMA: dict[str, type[pl.DataType]] = {name: pl.Utf8 for name in RAW_REDTEAM_COLUMNS}
RAW_REDTEAM_SCHEMA["time"] = pl.Int64

# --- Typed auth event, post `parse.clean`. ------------------------------------------
#
# `time` is Int32 (max LANL time ~5.01M seconds, well under the 2^31 bound).
# High-cardinality identifiers are Polars `Categorical` — never one-hot at this
# stage; individual model/feature code decides its own encoding downstream.
# The four sentinel/null indicator columns exist because nullity in this
# dataset is informative (~55% of `auth_type`, ~14% of `logon_type` are null)
# and must never be silently imputed.

TYPED_AUTH_SCHEMA: dict[str, type[pl.DataType]] = {
    "time": pl.Int32,
    "day": pl.Int16,
    "src_user": pl.Categorical,
    "src_domain": pl.Categorical,
    "dst_user": pl.Categorical,
    "dst_domain": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
    "auth_type": pl.Categorical,
    "auth_type_is_null": pl.Boolean,
    "logon_type": pl.Categorical,
    "logon_type_is_null": pl.Boolean,
    "auth_orientation": pl.Categorical,
    "success": pl.Boolean,
    "src_user_is_machine": pl.Boolean,
    "dst_user_is_machine": pl.Boolean,
    "event_id": pl.UInt64,
}

TYPED_REDTEAM_SCHEMA: dict[str, type[pl.DataType]] = {
    "time": pl.Int32,
    "day": pl.Int16,
    "user": pl.Categorical,
    "domain": pl.Categorical,
    "src_computer": pl.Categorical,
    "dst_computer": pl.Categorical,
}

# Nullable string sentinel LANL uses for missing auth_type / logon_type.
LANL_NULL_TOKEN = "?"

# Join key for exact label attribution (US-105) — the quadruplet, never time alone.
LABEL_JOIN_KEYS: list[str] = ["time", "user_at_domain", "src_computer", "dst_computer"]

SECONDS_PER_DAY = 86_400
