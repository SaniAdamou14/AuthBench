import resource, sys, time
import polars as pl
t0 = time.time()
auth = pl.scan_parquet("/root/AuthBench/data/interim/auth/**/*.parquet").select("time", "src_computer", "dst_computer")
def edges(frame):
    return (frame.select((pl.col("time") // 1800).alias("snapshot"), pl.col("src_computer").cast(pl.Utf8), pl.col("dst_computer").cast(pl.Utf8))
            .group_by("snapshot", "src_computer", "dst_computer").len().collect())
train = edges(auth.filter(pl.col("time") < 150885))
per = train.group_by("snapshot").len()
nodes = pl.concat([train["src_computer"], train["dst_computer"]]).n_unique()
print("train events", int(train["len"].sum()), "edges", train.height, "snapshots", per.height, "edges/snapshot max", per["len"].max(), "mean", round(per["len"].mean()), "nodes", nodes, flush=True)
for day in (5, 20, 40):
    d = edges(auth.filter(pl.col("time").is_between(day*86400, (day+1)*86400, closed="left")))
    print("day", day, "events", int(d["len"].sum()), "distinct edges", d.height, "estimated MB", round(d.estimated_size()/1e6), flush=True)
print("peak rss GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, 1), "seconds", round(time.time()-t0), flush=True)
