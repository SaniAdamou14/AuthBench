"""F1 — event-level features: zero cost, available immediately, no history needed (US-109).

The only F1 features that are not pure row-local transforms are the three
frequency encodings. A frequency is a property of a *distribution*, so it has
to be fitted somewhere — and fitting it on the frame being encoded is a
leak: it lets each event's encoding depend on events that come after it, and
it gives the same category a different numeric value in train and in test.
`fit_frequency_encoding` therefore learns the tables on the training split
only (US-107), and `compute_f1` applies those fixed tables to every split.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

F1_COLUMNS: list[str] = [
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

# Frequency encoding — not one-hot: `auth_type`/`logon_type`/`auth_orientation`
# are high-cardinality categoricals where one-hot would explode dimensionality.
FREQUENCY_ENCODED_COLUMNS: list[str] = ["auth_type", "logon_type", "auth_orientation"]

# A category absent from the training split has, by definition, a training
# frequency of zero. Encoding it as 0.0 is both the correct answer and the
# operationally meaningful one — "never seen while the model was learning" is
# the extreme of the same axis the encoding measures.
UNSEEN_CATEGORY_FREQUENCY = 0.0


@dataclass(frozen=True)
class FrequencyEncoding:
    """Fitted category -> relative-frequency tables, one per encoded column.

    `null_rates` is kept separately because null is a real, informative
    category in LANL (~55% of `auth_type`) but cannot be a dict key in a
    Polars `replace_strict` mapping.
    """

    tables: dict[str, dict[str, float]]
    null_rates: dict[str, float]

    def expr(self, column: str) -> pl.Expr:
        """The `<column>_freq` expression for this fitted encoding."""
        table = self.tables[column]
        as_string = pl.col(column).cast(pl.Utf8)
        known = (
            as_string.replace_strict(
                table, default=UNSEEN_CATEGORY_FREQUENCY, return_dtype=pl.Float64
            )
            if table
            else pl.lit(UNSEEN_CATEGORY_FREQUENCY, dtype=pl.Float64)
        )
        return (
            pl.when(pl.col(column).is_null())
            .then(pl.lit(self.null_rates[column], dtype=pl.Float64))
            .otherwise(known)
            .alias(f"{column}_freq")
        )


def fit_frequency_encoding(
    train: pl.LazyFrame, columns: list[str] | None = None
) -> FrequencyEncoding:
    """Learn each column's category frequencies from the training split only.

    Call once, on `get_train_split(...)`'s output, and pass the result to
    every `compute_f1` call — train, val and test alike. Fitting per split
    would make the same `auth_type` mean a different number in each of them.
    """
    columns = columns or FREQUENCY_ENCODED_COLUMNS

    n_rows = train.select(pl.len()).collect().item()
    if n_rows == 0:
        raise ValueError("Cannot fit a frequency encoding on an empty training split.")

    tables: dict[str, dict[str, float]] = {}
    null_rates: dict[str, float] = {}
    for column in columns:
        counts = (
            train.group_by(column).agg(pl.len().alias("_n")).collect().sort(column, nulls_last=True)
        )
        table: dict[str, float] = {}
        null_rate = 0.0
        for category, n in zip(
            counts[column].cast(pl.Utf8).to_list(), counts["_n"].to_list(), strict=True
        ):
            if category is None:
                null_rate = n / n_rows
            else:
                table[category] = n / n_rows
        tables[column] = table
        null_rates[column] = null_rate

    return FrequencyEncoding(tables=tables, null_rates=null_rates)


def compute_f1(events: pl.LazyFrame, encoding: FrequencyEncoding) -> pl.LazyFrame:
    """Add the F1 event-level feature columns to a typed auth-event frame.

    `encoding` must come from `fit_frequency_encoding` on the *training*
    split. It is a required argument rather than an optional one on purpose:
    the previous signature made the leaky behaviour the default, and a
    leak-free protocol that depends on the caller remembering a keyword is
    not a protocol.
    """
    return events.with_columns(
        [
            pl.col("success").alias("is_success"),
            pl.col("auth_type_is_null"),
            pl.col("logon_type_is_null"),
            pl.col("src_user_is_machine"),
            (pl.col("src_user") == pl.col("dst_user")).alias("src_dst_user_same"),
            (pl.col("src_computer") == pl.col("dst_computer")).alias("src_dst_computer_same"),
            (pl.col("src_domain") != pl.col("dst_domain")).alias("domain_crossing"),
            *[encoding.expr(column) for column in FREQUENCY_ENCODED_COLUMNS],
        ]
    )
