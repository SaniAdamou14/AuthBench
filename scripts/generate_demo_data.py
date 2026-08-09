#!/usr/bin/env python
"""Deterministic synthetic demo dataset generator (US-103).

Produces a small, LANL-shaped `auth.txt`/`redteam.txt` pair under
`data/demo/`, small enough to run the entire pipeline in well under five
minutes (NFR-04), while preserving the properties that make the pipeline
meaningful to demo:

- normal per-user behavior (a small set of "home" machines, a preferred
  login-hour window, some jitter) so history/novelty/temporal features have
  something real to key on;
- machine accounts (`$` suffix) with their own distinct pattern;
- realistic null rates for `auth_type` (~55%) and `logon_type` (~14%);
- three complete red-team lateral-movement campaigns (A→B→C chains), one in
  each of the train/val/test partitions of `conf/split/demo_temporal.yaml`,
  with every campaign event's quadruplet duplicated verbatim into the
  redteam file, and the full history of the users/machines involved
  preserved — a naive random event sample would destroy exactly the
  historical features this benchmark is about (US-103 acceptance criteria).

Deterministic: same seed, same output, byte for byte.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np

DOMAIN = "DOM1"
AUTH_TYPES = ["Kerberos", "NTLM", "Negotiate", "LM"]
LOGON_TYPES = ["Network", "Interactive", "Batch", "Service"]
ORIENTATIONS = ["LogOn", "LogOff"]
NULL_TOKEN = "?"

AUTH_TYPE_NULL_RATE = 0.55
LOGON_TYPE_NULL_RATE = 0.14
FAILURE_RATE = 0.03


def _fmt_row(*fields: object) -> str:
    return ",".join(str(f) for f in fields)


def generate(
    *,
    n_days: int,
    n_users: int,
    n_machine_accounts: int,
    n_computers: int,
    events_per_user_per_day: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    seconds_per_day = 86_400

    users = [f"U{idx:04d}" for idx in range(n_users)]
    machine_accounts = [f"M{idx:04d}$" for idx in range(n_machine_accounts)]
    computers = [f"C{idx:04d}" for idx in range(n_computers)]

    # Each human user gets a small "home" set of machines and a preferred
    # login-hour window, so F2/F3/F4 have real signal to find.
    user_home_computers = {
        u: rng.choice(computers, size=rng.integers(2, 5), replace=False).tolist() for u in users
    }
    user_preferred_hour = {u: int(rng.integers(6, 20)) for u in users}
    user_src_computer = {u: rng.choice(computers) for u in users}

    auth_lines: list[str] = []
    redteam_lines: list[str] = []

    def emit(
        time: int,
        src_user: str,
        dst_user: str,
        src_computer: str,
        dst_computer: str,
        auth_type: str,
        logon_type: str,
        orientation: str,
        success: bool,
    ) -> None:
        at = NULL_TOKEN if rng.random() < AUTH_TYPE_NULL_RATE else auth_type
        lt = NULL_TOKEN if rng.random() < LOGON_TYPE_NULL_RATE else logon_type
        auth_lines.append(
            _fmt_row(
                time,
                f"{src_user}@{DOMAIN}",
                f"{dst_user}@{DOMAIN}",
                src_computer,
                dst_computer,
                at,
                lt,
                orientation,
                "Success" if success else "Failure",
            )
        )

    # --- Normal human traffic -------------------------------------------------
    for day in range(n_days):
        day_start = day * seconds_per_day
        for u in users:
            n_events = rng.poisson(events_per_user_per_day)
            preferred_hour = user_preferred_hour[u]
            for _ in range(n_events):
                hour = int(np.clip(rng.normal(preferred_hour, 1.5), 0, 23)) % 24
                second_of_day = hour * 3600 + int(rng.integers(0, 3600))
                time = day_start + second_of_day
                dst = (
                    rng.choice(user_home_computers[u])
                    if rng.random() > 0.05
                    else rng.choice(computers)
                )
                emit(
                    time,
                    u,
                    u,
                    user_src_computer[u],
                    dst,
                    rng.choice(AUTH_TYPES),
                    rng.choice(LOGON_TYPES),
                    rng.choice(ORIENTATIONS),
                    rng.random() > FAILURE_RATE,
                )

    # --- Machine-account traffic: frequent, bursty, distinct pattern ----------
    for day in range(n_days):
        day_start = day * seconds_per_day
        for m in machine_accounts:
            n_events = rng.poisson(events_per_user_per_day * 3)
            for _ in range(n_events):
                time = day_start + int(rng.integers(0, seconds_per_day))
                dst = rng.choice(computers)
                emit(
                    time,
                    m,
                    m,
                    rng.choice(computers),
                    dst,
                    rng.choice(AUTH_TYPES),
                    "Service",
                    "LogOn",
                    True,
                )

    # --- Red-team campaigns: three complete lateral-movement chains -----------
    # One campaign per partition of conf/split/demo_temporal.yaml (train
    # [0, 7], val [8, 10], test [11, 13] at the default n_days=14). Every
    # partition needs its own positives, for a different reason each time:
    #
    #   train — so the semi-supervised regime (r2) has anything to learn from;
    #   val   — M1's weight calibration maximizes AUC-PR on validation only.
    #           With zero positives there, average_precision_score returns
    #           0.0 for *every* trial, so the random search silently keeps its
    #           first arbitrary draw and "calibration" becomes a no-op;
    #   test  — otherwise every model reports AUC-PR 0 and the demo looks
    #           broken for the wrong reason.
    #
    # The day offsets are expressed as fractions of n_days so that a demo
    # regenerated with a different --n-days keeps landing one campaign in
    # each partition.
    campaign_specs = [
        {"user": users[0], "day": max(2, n_days // 4), "chain": computers[:4], "start_hour": 3},
        {
            "user": users[1],
            "day": max(3, int(n_days * 0.64)),
            "chain": computers[4:8],
            "start_hour": 5,
        },
        {"user": users[2], "day": max(4, n_days - 2), "chain": computers[8:12], "start_hour": 2},
    ]

    for spec in campaign_specs:
        u = spec["user"]
        day_start = spec["day"] * seconds_per_day
        t = day_start + spec["start_hour"] * 3600
        chain = spec["chain"]
        for hop in range(len(chain) - 1):
            src_c, dst_c = chain[hop], chain[hop + 1]
            t += int(rng.integers(60, 600))  # each hop a few minutes after the last
            auth_lines.append(
                _fmt_row(
                    t,
                    f"{u}@{DOMAIN}",
                    f"{u}@{DOMAIN}",
                    src_c,
                    dst_c,
                    rng.choice(AUTH_TYPES),
                    rng.choice(LOGON_TYPES),
                    "LogOn",
                    "Success",
                )
            )
            redteam_lines.append(_fmt_row(t, f"{u}@{DOMAIN}", src_c, dst_c))

    return auth_lines, redteam_lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--n-days", type=int, default=14)
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--n-machine-accounts", type=int, default=50)
    parser.add_argument("--n-computers", type=int, default=100)
    parser.add_argument("--events-per-user-per-day", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gzip", action="store_true", help="Write .gz to mirror LANL's own format."
    )
    args = parser.parse_args()

    auth_lines, redteam_lines = generate(
        n_days=args.n_days,
        n_users=args.n_users,
        n_machine_accounts=args.n_machine_accounts,
        n_computers=args.n_computers,
        events_per_user_per_day=args.events_per_user_per_day,
        seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.gzip else open
    suffix = ".gz" if args.gzip else ""

    auth_path = args.out_dir / f"auth_demo.txt{suffix}"
    redteam_path = args.out_dir / f"redteam_demo.txt{suffix}"

    with opener(auth_path, "wt") as f:
        f.write("\n".join(auth_lines) + "\n")
    with opener(redteam_path, "wt") as f:
        f.write("\n".join(redteam_lines) + "\n")

    print(f"Wrote {len(auth_lines):,} auth events to {auth_path}")
    print(f"Wrote {len(redteam_lines):,} red-team events to {redteam_path}")


if __name__ == "__main__":
    main()
