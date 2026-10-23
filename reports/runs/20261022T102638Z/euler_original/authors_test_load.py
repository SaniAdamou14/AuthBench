"""Count what the authors' own test loader hands their workers: run.py -w 4 tests on
(150885, 5011199) with delta 1800, through spinup.get_work_units and load_lanl_dist."""

import sys

sys.path.insert(0, "/root/euler_original/lanl_experiments")
from spinup import get_work_units  # noqa: E402
from loaders.load_lanl import load_lanl_dist  # noqa: E402

total_edges = total_pos = total_snaps = 0
first_snapshot_pos = None
for kw in get_work_units(4, 150885, 5011199, 1800, True):
    jobs = kw.pop("jobs")
    start, end = kw["start"], kw["end"]
    data = load_lanl_dist(jobs, **kw)
    n_edges = sum(int(ei.size(1)) for ei in data.eis)
    n_pos = sum(int(y.sum()) for y in data.ys)
    if first_snapshot_pos is None:
        first_snapshot_pos = int(data.ys[0].sum())
    print(f"worker {start}-{end}: {len(data.eis)} snapshots, {n_edges} edges, {n_pos} anomalous", flush=True)
    total_edges += n_edges
    total_pos += n_pos
    total_snaps += len(data.eis)
print(f"total: {total_snaps} snapshots, {total_edges} edges, {total_pos} anomalous")
print(f"anomalous in the first test snapshot (never scored, decode_all starts at 1 on the head): {first_snapshot_pos}")
