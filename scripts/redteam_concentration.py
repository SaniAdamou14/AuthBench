"""Concentration of the LANL red-team events on their source computers, per time window.

Reads redteam.txt (plain or gzipped) and writes only aggregates: for each window, the number of
distinct red-team events, of source computers and of accounts, and the events of the most frequent
source computers. The harmonised protocol's run outputs carry no source computer, so this is how
the paper states its concentration; it counts red-team rows, not the labelled authentication
events (a row without a matching authentication, or matching two, makes the counts differ).

    python scripts/redteam_concentration.py data/raw/redteam.txt.gz out.json
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HOUR = 3600
DAY = 86400
# (name, first second included, last second excluded)
WINDOWS = [
    ("all", 0, None),
    ("harmonised_test", 41 * HOUR, 1392 * HOUR),
    ("temporal_test", 11 * DAY, 14 * DAY),
]


def read_rows(path: Path) -> list[tuple[int, str, str, str]]:
    raw = path.read_bytes()
    text = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    rows = set()
    for line in text.decode().splitlines():
        if line.strip():
            time, user, src, dst = line.split(",")
            rows.add((int(time), user, src, dst))
    return sorted(rows)


def summary(rows: list[tuple[int, str, str, str]]) -> dict[str, object]:
    sources = Counter(src for _, _, src, _ in rows)
    return {
        "events": len(rows),
        "accounts": len({user for _, user, _, _ in rows}),
        "source_computers": len(sources),
        "top_sources": [
            {"src_computer": src, "events": n, "share": n / len(rows)}
            for src, n in sources.most_common(3)
        ],
    }


def main(source: Path, out: Path) -> None:
    rows = read_rows(source)
    result: dict[str, object] = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "distinct_rows": len(rows),
    }
    for name, first, last in WINDOWS:
        kept = [r for r in rows if r[0] >= first and (last is None or r[0] < last)]
        result[name] = summary(kept)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
