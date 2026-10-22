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
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "euler_fidelity.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("euler_fidelity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network(tmp_path: Path) -> tuple[Path, Path, pl.DataFrame]:
    """Three days of a small network; the red team strikes twice on day 1 and once on day 2."""
    rng = np.random.default_rng(0)
    n = 6_000
    time = np.sort(rng.integers(0, 3 * 86_400, n))
    src = [f"C{i}" for i in rng.integers(0, 12, n)]
    dst = [f"C{i}" for i in rng.integers(0, 12, n)]
    attacks = [(90_000, "U1@DOM1", "C3", "C40"), (90_050, "U1@DOM1", "C3", "C40")]
    attacks.append((200_000, "U2@DOM1", "C5", "C7"))
    for t, _, s, d in attacks:
        time = np.append(time, t)
        src.append(s)
        dst.append(d)
    auth = pl.DataFrame({"time": time, "src_computer": src, "dst_computer": dst}).sort("time")
    for day in range(3):
        directory = tmp_path / "auth" / f"day={day}"
        directory.mkdir(parents=True)
        auth.filter((pl.col("time") // 86_400) == day).write_parquet(directory / "part.parquet")
    redteam = tmp_path / "redteam.txt"
    redteam.write_text("".join(f"{t},{u},{s},{d}\n" for t, u, s, d in attacks))
    return tmp_path / "auth", redteam, auth


def test_the_day_by_day_check_equals_one_computation_over_every_edge(tmp_path: Path) -> None:
    script = _script()
    auth_dir, redteam, auth = _network(tmp_path)
    model = EulerScorer(epochs=3, patience=2, window=10, seed=1)
    result = script.run(auth_dir, redteam, tmp_path / "out", model)
    assert json.loads((tmp_path / "out" / "euler_fidelity.json").read_text()) == result

    first_attack = 90_000
    edges = (
        auth.select((pl.col("time") // SNAPSHOT).alias("snapshot"), "src_computer", "dst_computer")
        .unique()
        .filter(pl.col("snapshot") * SNAPSHOT >= first_attack)
    )
    attacked = {(90_000 // SNAPSHOT, "C3", "C40"), (200_000 // SNAPSHOT, "C5", "C7")}
    labels = np.array([tuple(row) in attacked for row in edges.iter_rows()])
    scores = model.score(
        edges.select(
            (pl.col("snapshot") * SNAPSHOT).alias("time"), "src_computer", "dst_computer"
        ).lazy()
    ).to_numpy()
    assert result["first_red_team_event"] == first_attack
    assert result["test_edges"] == edges.height
    assert result["anomalous_edges"] == 2
    assert result["auc"] == pytest.approx(roc_auc_score(labels, scores), rel=1e-12)
    assert result["ap"] == pytest.approx(average_precision_score(labels, scores), rel=1e-12)
