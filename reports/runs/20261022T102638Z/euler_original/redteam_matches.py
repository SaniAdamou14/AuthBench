"""For every red-team event: is there an authentication event at the same second with the same
source and destination computers (any type), and an NTLM one? Red-team edges without a matching
event cannot be labelled by AuthBench (join on observed edges) nor by the authors (exact match)."""

import polars as pl

red = pl.read_csv("/root/euler_original_data/redteam.txt", has_header=False,
                  new_columns=["time", "user", "src", "dst"]).unique()
auth = pl.scan_parquet("/root/AuthBench/data/interim/auth/*/*.parquet")
print(auth.collect_schema())
times = red["time"].unique().to_list()
hits = (
    auth.filter(pl.col("time").is_in(times))
    .select(pl.col("time").cast(pl.Int64), pl.col("src_computer").cast(pl.Utf8).alias("src"),
            pl.col("dst_computer").cast(pl.Utf8).alias("dst"),
            pl.col("auth_type").cast(pl.Utf8).str.to_uppercase().str.contains("NTLM").fill_null(False).alias("ntlm"))
    .group_by("time", "src", "dst").agg(pl.len().alias("n"), pl.col("ntlm").any())
    .collect()
)
m = red.with_columns(pl.col("time").cast(pl.Int64)).join(hits, on=["time", "src", "dst"], how="left")
print("red-team rows (distinct):", m.height, " with any matching event:", m["n"].is_not_null().sum(),
      " with an NTLM one:", m["ntlm"].fill_null(False).sum(), " self-loops:", (m["src"] == m["dst"]).sum())
print("without a matching event:")
with pl.Config(tbl_rows=50):
    print(m.filter(pl.col("n").is_null()).sort("time"))
    print("matched but not NTLM:")
    print(m.filter(pl.col("n").is_not_null() & ~pl.col("ntlm")).sort("time"))
