import time
import polars as pl
t0 = time.time()
auth = pl.scan_parquet("/root/AuthBench/data/interim/auth/**/*.parquet")
cols = auth.collect_schema().names()
red = (pl.scan_csv("/root/AuthBench/data/raw/redteam.txt.gz", has_header=False, new_columns=["time","user","src","dst"])
       .select((pl.col("time")//1800).alias("snapshot"), "src", "dst").unique().with_columns(pl.lit(True).alias("bad")).collect())
text = pl.concat_str([pl.col(c).cast(pl.Utf8).fill_null("") for c in cols if c not in ("time",)], separator=",").str.to_uppercase()
counts = {k: [0, 0] for k in ("all", "no_self_loop", "ntlm", "ntlm_no_self_loop")}
for day in range(1, 59):
    f = auth.filter(pl.col("time").is_between(max(day*86400, 150885), (day+1)*86400, closed="left"))
    e = (f.select((pl.col("time")//1800).alias("snapshot"), pl.col("src_computer").cast(pl.Utf8).alias("src"), pl.col("dst_computer").cast(pl.Utf8).alias("dst"),
                  text.str.contains("NTLM").alias("ntlm"))
          .group_by("snapshot", "src", "dst").agg(pl.col("ntlm").any()).collect()
          .join(red, on=["snapshot", "src", "dst"], how="left").with_columns(pl.col("bad").fill_null(False)))
    loop = e["src"] == e["dst"]
    for k, m in (("all", None), ("no_self_loop", ~loop), ("ntlm", e["ntlm"]), ("ntlm_no_self_loop", e["ntlm"] & ~loop)):
        s = e if m is None else e.filter(m)
        counts[k][0] += s.height; counts[k][1] += int(s["bad"].sum())
for k, (n, b) in counts.items():
    print(f"{k}: {n} edges, {b} anomalous, base rate {b/n:.3e}", flush=True)
print("columns used for the NTLM match:", [c for c in cols if c != "time"], "seconds", round(time.time()-t0), flush=True)
