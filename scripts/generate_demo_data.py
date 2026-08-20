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
- ten complete red-team lateral-movement campaigns (A→B→C chains) spread
  across the train/val/test partitions of `conf/split/demo_temporal.yaml`
  — three, three and **four** respectively, for the reasons set out at
  `_CAMPAIGNS_PER_PARTITION` — with every campaign event's quadruplet
  duplicated verbatim into the redteam file, and the full history of the
  users/machines involved preserved: a naive random event sample would
  destroy exactly the historical features this benchmark is about (US-103
  acceptance criteria).

Deterministic: same seed, same output, byte for byte.
"""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CampaignSpec:
    """One red-team lateral-movement chain: a user walking a path of machines."""

    user: str
    day: int
    start_hour: int
    chain: list[str]


# Where each partition of `conf/split/demo_temporal.yaml` sits, as a fraction of
# `n_days`, so a sample regenerated with a different `--n-days` keeps landing
# campaigns in the right places. At the default 14 days these resolve to the
# configured train [0, 7] / val [8, 10] / test [11, 13].
#
# Day 0 is deliberately excluded: an attack on the first day has no prior
# history for F2/F3 to contrast it against, so it would be detected (or missed)
# for reasons that have nothing to do with the models.
_PARTITION_DAY_FRACTIONS: dict[str, tuple[float, float]] = {
    "train": (0.15, 0.54),
    "val": (0.58, 0.76),
    "test": (0.80, 0.97),
}

# How many campaigns land in each partition, and why the counts are not equal.
#
# **test gets the most, and never fewer than two.** The number of campaigns in
# a partition *is* the sample size of the campaign-stratified bootstrap that
# produces every confidence interval and every pairwise p-value. With one
# campaign in the test split, resampling cannot vary the campaign composition
# at all: no pairwise difference can change sign, every p-value collapses onto
# the resolution floor 2/(R+1), and the whole comparison table reads
# "significant" off a single attack. That is a property of the sample, not of
# the models, and `stats_tests.MIN_CAMPAIGNS_FOR_SIGNIFICANCE` now refuses to
# report a verdict below two — so a one-campaign test split makes the demo
# unable to demonstrate the one thing it exists to demonstrate.
#
# **val gets three**, because M1's weight calibration maximizes AUC-PR there
# and a single campaign makes that objective nearly degenerate.
#
# **train gets three**, so the semi-supervised regime (R2) has something to
# learn from and the unsupervised regime (R1) has contamination to be robust to.
_CAMPAIGNS_PER_PARTITION: dict[str, int] = {"train": 3, "val": 3, "test": 4}

# Chain lengths cycle, so campaigns differ in event count the way real ones do
# — a two-hop smash-and-grab and a four-hop walk are not the same detection
# problem, and `campaign_summary.csv` should show that.
_CHAIN_LENGTHS: list[int] = [3, 4, 3, 5]

# Most intrusions here happen at night, where F4's calibrated night window can
# see them. Two do not: a benchmark whose every positive is trivially separable
# on one feature measures that feature, not the models.
_NIGHT_START_HOURS: list[int] = [2, 3, 4, 5]
_DAYTIME_CAMPAIGN_INDICES: frozenset[int] = frozenset({4, 8})


def build_campaign_specs(n_days: int, users: list[str], computers: list[str]) -> list[CampaignSpec]:
    """Lay out the red-team campaigns across the three partitions.

    Every campaign gets its own user, because `label.redteam_join` groups
    campaigns per `user@domain`: two chains sharing a user within the 24-hour
    gap threshold would be merged into one campaign, silently undoing the
    counts above.
    """
    specs: list[CampaignSpec] = []
    index = 0

    for partition, n_campaigns in _CAMPAIGNS_PER_PARTITION.items():
        low_fraction, high_fraction = _PARTITION_DAY_FRACTIONS[partition]
        first_day = max(1, int(n_days * low_fraction))
        last_day = max(first_day, int(n_days * high_fraction))
        span = last_day - first_day + 1

        for slot in range(n_campaigns):
            chain_length = _CHAIN_LENGTHS[index % len(_CHAIN_LENGTHS)]
            chain_start = (index * 5) % max(1, len(computers) - chain_length)
            specs.append(
                CampaignSpec(
                    user=users[index % len(users)],
                    # Spread across the partition's days; two campaigns may
                    # share a day when the partition is shorter than the
                    # number of campaigns, which is fine — different users
                    # keep them separate campaigns.
                    day=first_day + (slot % span),
                    start_hour=(
                        10 + (index % 5)
                        if index in _DAYTIME_CAMPAIGN_INDICES
                        else _NIGHT_START_HOURS[index % len(_NIGHT_START_HOURS)]
                    ),
                    chain=computers[chain_start : chain_start + chain_length],
                )
            )
            index += 1

    return specs


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

    # --- Red-team campaigns: complete lateral-movement chains ------------------
    for spec in build_campaign_specs(n_days, users, computers):
        u = spec.user
        t = spec.day * seconds_per_day + spec.start_hour * 3600
        for hop in range(len(spec.chain) - 1):
            src_c, dst_c = spec.chain[hop], spec.chain[hop + 1]
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
