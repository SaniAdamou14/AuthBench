"""The fidelity check, built and scored a day at a time, equals one computation over every edge."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

pytest.importorskip("torch")

from authbench.models.euler import EulerScorer  # noqa: E402

SNAPSHOT = 1800
FIRST_ATTACK = 90_900  # inside the snapshot [90000, 91800)
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "euler_fidelity.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("euler_fidelity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network(tmp_path: Path) -> tuple[Path, Path, pl.DataFrame]:
    """Three days of a small network, NTLM and Kerberos, self-loops included; the red team
    strikes over NTLM twice on day 1 (once inside the snapshot of its first event) and once on
    day 2."""
    rng = np.random.default_rng(0)
    n = 6_000
    time = np.sort(rng.integers(0, 3 * 86_400, n))
    src = [f"C{i}" for i in rng.integers(0, 12, n)]
    dst = [f"C{i}" for i in rng.integers(0, 12, n)]
    auth_type = list(rng.choice(["NTLM", "Kerberos", "Negotiate"], n))
    attacks = [(FIRST_ATTACK, "U1@DOM1", "C3", "C40"), (FIRST_ATTACK + 50, "U1@DOM1", "C3", "C41")]
    attacks.append((200_000, "U2@DOM1", "C5", "C7"))
    # Earlier in the first attack's snapshot: a test edge only from the first attack on.
    time = np.append(time, [FIRST_ATTACK - 30])
    src.append("C9")
    dst.append("C10")
    auth_type.append("NTLM")
    for t, _, s, d in attacks:
        time = np.append(time, t)
        src.append(s)
        dst.append(d)
        auth_type.append("NTLM")
    auth = pl.DataFrame(
        {"time": time, "src_computer": src, "dst_computer": dst, "auth_type": auth_type}
    ).sort("time")
    for day in range(3):
        directory = tmp_path / "auth" / f"day={day}"
        directory.mkdir(parents=True)
        auth.filter((pl.col("time") // 86_400) == day).write_parquet(directory / "part.parquet")
    redteam = tmp_path / "redteam.txt"
    redteam.write_text("".join(f"{t},{u},{s},{d}\n" for t, u, s, d in attacks))
    return tmp_path / "auth", redteam, auth


QUICK = {"epochs": 3, "patience": 2, "window": 10}


@pytest.mark.parametrize("config", ["all", "ntlm", "ntlm_vocabulary"])
def test_the_day_by_day_check_equals_one_computation_over_every_edge(
    tmp_path: Path, config: str
) -> None:
    script = _script()
    auth_dir, redteam, auth = _network(tmp_path)
    result = script.run(auth_dir, redteam, tmp_path / "out", config, (1, 2), **QUICK)
    written = tmp_path / "out" / f"euler_fidelity_{config}.json"
    assert json.loads(written.read_text()) == result

    if config != "all":
        auth = auth.filter(
            (pl.col("auth_type") == "NTLM") & (pl.col("src_computer") != pl.col("dst_computer"))
        )
    events = auth.select("time", "src_computer", "dst_computer")
    vocabulary = None
    if config == "ntlm_vocabulary":
        vocabulary = sorted(set(events["src_computer"]) | set(events["dst_computer"]))
    edges = (
        events.filter(pl.col("time") >= FIRST_ATTACK)
        .select((pl.col("time") // SNAPSHOT).alias("snapshot"), "src_computer", "dst_computer")
        .unique()
    )
    attacked = {
        (FIRST_ATTACK // SNAPSHOT, "C3", "C40"),
        ((FIRST_ATTACK + 50) // SNAPSHOT, "C3", "C41"),
        (200_000 // SNAPSHOT, "C5", "C7"),
    }
    labels = np.array([tuple(row) in attacked for row in edges.iter_rows()])
    assert result["config"] == config
    assert result["first_red_team_event"] == FIRST_ATTACK
    assert result["test_edges"] == edges.height
    assert result["anomalous_edges"] == 3
    assert result["events_kept"] == events.height
    for seed, run in zip((1, 2), result["runs"], strict=True):
        model = EulerScorer(**QUICK, seed=seed, vocabulary=vocabulary)
        model.fit(events.filter(pl.col("time") < FIRST_ATTACK).lazy())
        model.prepare([events.lazy()])
        scores = model.score(
            edges.select(
                (pl.col("snapshot") * SNAPSHOT).alias("time"), "src_computer", "dst_computer"
            ).lazy()
        ).to_numpy()
        assert run["seed"] == seed
        assert run["auc"] == pytest.approx(roc_auc_score(labels, scores), rel=1e-12)
        assert run["ap"] == pytest.approx(average_precision_score(labels, scores), rel=1e-12)
    assert result["auc_mean"] == pytest.approx(np.mean([r["auc"] for r in result["runs"]]))
    assert result["ap_std"] == pytest.approx(np.std([r["ap"] for r in result["runs"]], ddof=1))


def test_the_code_configuration_takes_the_authors_defaults_under_any_override(
    tmp_path: Path,
) -> None:
    script = _script()
    auth_dir, redteam, _ = _network(tmp_path)
    result = script.run(auth_dir, redteam, tmp_path / "out", "ntlm_code", (1,), epochs=2)
    assert result["params"] == {"learning_rate": 0.005, "patience": 5, "epochs": 2}
    assert result["vocabulary"] == "every node of the timeline"


def test_the_ntlm_variant_keeps_ntlm_events_between_two_computers() -> None:
    script = _script()
    auth = pl.LazyFrame(
        {
            "time": [1, 2, 3, 4],
            "src_computer": ["C1", "C1", "C2", "C3"],
            "dst_computer": ["C2", "C1", "C3", "C4"],
            "auth_type": ["NTLM", "NTLM", "Kerberos", None],
            "logon_type": ["Network", "Network", "Network", "Ntlm-ish"],
        }
    )
    kept = script.select_events(auth, "ntlm").collect()
    # The authors' filter is a case-insensitive search of the whole line, not of auth_type.
    assert kept["time"].to_list() == [1, 4]
    assert script.select_events(auth, "all").collect().height == 4
    with pytest.raises(ValueError, match="Unknown variant"):
        script.select_events(auth, "kerberos")
