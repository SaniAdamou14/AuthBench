"""Row count and an order-independent content fingerprint of converted LANL days."""

import sys
from pathlib import Path

import polars as pl

root = Path(sys.argv[1])
for day in [int(d) for d in sys.argv[2].split(",")]:
    frame = pl.read_parquet(root / f"day={day}" / "*.parquet")
    text = frame.select(pl.all().cast(pl.Utf8))
    digest = int(text.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3).cast(pl.UInt64).sum()) % (1 << 64)
    print(day, frame.height, digest)
