"""Generate every number, table and figure of the paper from the committed run outputs.

Writes `reports/paper/generated/`: `numbers.tex` (macros), one `.tex` table per
table, and the figures. Nothing in the paper is typed by hand; CI regenerates
the `.tex` files and fails if they differ from the committed copies.

The evaluated run is `reports/runs/20261022T102638Z` (both protocols, Euler's fidelity check
and the authors' Euler code). The wider-window run of September 2026 is read only for the
temporal split statistics, which its `split_summary.json` records for the same partitions,
and for the comparison with the earlier code; the demo for the circularity control.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from authbench.evaluate.budget import (  # noqa: E402
    random_first_detection_quantile,
    random_first_detection_survival,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports" / "runs" / "20261022T102638Z"
WIDE = ROOT / "reports" / "lanl" / "wider_window_results"
DEMO = ROOT / "reports" / "demo" / "tables"
OUT = ROOT / "reports" / "paper" / "generated"

PROTOCOLS = {
    "Harm": (RUN / "harmonised" / "metrics_summary.json", RUN / "harmonised" / "tables"),
    "Temp": (RUN / "temporal" / "metrics_summary.json", RUN / "temporal" / "tables"),
    "TempTwo": (
        RUN / "temporal" / "r2_semi_supervised" / "metrics_summary.json",
        RUN / "temporal" / "tables" / "r2_semi_supervised",
    ),
}

LABELS = {
    "M0a_random": "M0a random",
    "M0b_always_fail": "M0b always-fail",
    "M1_rules": "M1 rules",
    "M1u_rules_uniform": "M1u rules, uniform",
    "M2a_pair_rarity": "M2a pair rarity",
    "M2c_source_rarity": "M2c source rarity",
    "M2b_pca_reconstruction": "M2b PCA",
    "M3a_iforest": "M3a Isolation Forest",
    "M3b_hbos": "M3b HBOS",
    "M3b_ecod": "M3c ECOD",
    "M4a_autoencoder": "M4a autoencoder",
    "M4b_vae": "M4b VAE",
    "M5c_euler": "M5c Euler",
}
CODES = {
    "M0a_random": "M0a",
    "M0b_always_fail": "M0b",
    "M1_rules": "M1",
    "M1u_rules_uniform": "M1u",
    "M2a_pair_rarity": "M2a",
    "M2c_source_rarity": "M2c",
    "M2b_pca_reconstruction": "M2b",
    "M3a_iforest": "M3a",
    "M3b_hbos": "M3b",
    "M3b_ecod": "M3c",
    "M4a_autoencoder": "M4a",
    "M4b_vae": "M4b",
    "M5c_euler": "M5c",
}
MACRO_NAMES = {
    "M0a_random": "Random",
    "M0b_always_fail": "AlwaysFail",
    "M1_rules": "Rules",
    "M1u_rules_uniform": "RulesUniform",
    "M2a_pair_rarity": "PairRarity",
    "M2c_source_rarity": "SourceRarity",
    "M2b_pca_reconstruction": "Pca",
    "M3a_iforest": "IForest",
    "M3b_hbos": "Hbos",
    "M3b_ecod": "Ecod",
    "M4a_autoencoder": "Autoencoder",
    "M4b_vae": "Vae",
    "M5c_euler": "Euler",
}
ORDER = list(LABELS)
FLOORS = ("M0a_random", "M0b_always_fail")
BUDGET_NAMES = {10: "Ten", 50: "Fifty", 100: "Hundred", 500: "FiveHundred"}
METRICS = {
    "roc_auc": ("RocAuc", "ROC-AUC"),
    "auc_pr": ("AucPr", "AUC-PR"),
    "recall_fpr": ("RecallFpr", r"Recall at FPR $10^{-4}$"),
}
FPR_KEY = "0.0001"
PERMUTATIONS = 200_000
NULL_SIMULATIONS = 10_000
RULES = [
    ("R1_new_pair", "R1 new (user, destination) pair"),
    ("R2_host_burst", "R2 burst of destinations"),
    ("R3_fail_then_success", "R3 failures then success"),
    ("R4_offhours", "R4 unusual hour"),
    ("R5_machine_account_anomaly", "R5 machine-account rarity"),
    ("R6_new_auth_type", "R6 new authentication type"),
    ("R7_lateral_chain", "R7 lateral chain"),
    ("R8_new_source", "R8 new source computer"),
]
EULER_CONFIGS = [
    ("all", "AuthBench, all events"),
    ("ntlm", "AuthBench, NTLM only"),
    ("ntlm_vocabulary", "AuthBench, NTLM, authors' node vocabulary"),
    ("ntlm_code", "AuthBench, NTLM, vocabulary and authors' defaults"),
]
# King and Huang (2022): table V (events, anomalous edges) and table VI (GCN + GRU, link
# prediction). External published values, not run outputs.
EULER_PUBLISHED = {"events": 45_871_390, "anomalous": 518, "auc": 0.9906, "ap": 0.0155}
# Label offsets in points for the first-detection figure, per (protocol, metric), chosen so no
# label covers a point or another label; a leader line is drawn when the label is moved far.
LABEL_OFFSETS: dict[tuple[str, str], dict[str, tuple[int, int]]] = {
    ("Harm", "roc_auc"): {
        "M2c_source_rarity": (-20, 4),
        "M3b_ecod": (14, 8),
        "M3b_hbos": (14, -2),
        "M3a_iforest": (14, -10),
        "M2a_pair_rarity": (-22, -2),
        "M5c_euler": (14, -8),
        "M4a_autoencoder": (-22, 2),
        "M4b_vae": (14, -4),
        "M2b_pca_reconstruction": (-22, -4),
    },
    ("Harm", "auc_pr"): {
        "M3b_hbos": (-20, 2),
        "M3a_iforest": (4, -8),
        "M3b_ecod": (4, 3),
    },
    ("Temp", "roc_auc"): {
        "M3b_hbos": (-24, 8),
        "M2c_source_rarity": (-24, -6),
        "M2a_pair_rarity": (-8, 13),
        "M3b_ecod": (14, -14),
        "M4b_vae": (16, -3),
        "M5c_euler": (14, 8),
        "M3a_iforest": (14, 2),
        "M4a_autoencoder": (14, 2),
        "M2b_pca_reconstruction": (-22, -2),
    },
    ("Temp", "auc_pr"): {
        "M5c_euler": (2, 9),
        "M4b_vae": (8, -10),
        "M2a_pair_rarity": (-12, 12),
        "M3b_ecod": (-6, -13),
        "M2c_source_rarity": (-6, -13),
        "M3b_hbos": (-6, 8),
        "M4a_autoencoder": (4, 4),
        "M2b_pca_reconstruction": (4, -9),
    },
}
# Highlighted in the campaign-recall figure; every other non-floor detector is drawn in grey.
CURVE_HIGHLIGHT = {
    "Harm": ["M2b_pca_reconstruction", "M4b_vae", "M5c_euler", "M2a_pair_rarity"],
    "Temp": ["M1_rules", "M2b_pca_reconstruction", "M4a_autoencoder", "M5c_euler"],
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in rows}


def _int(value: float) -> str:
    return f"{round(value):,}".replace(",", "{,}")


def _sci(value: float, digits: int = 2) -> str:
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    return f"{value / 10**exponent:.{digits}f} \\times 10^{{{exponent}}}"


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}\\%"


def _p(value: float) -> str:
    return f"{value:.3f}"


class Macros:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.names: set[str] = set()

    def add(self, name: str, value: str) -> None:
        if name in self.names or not name.isalpha():
            raise ValueError(f"Bad or duplicate macro name: {name}")
        self.names.add(name)
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def write(self, path: Path) -> None:
        header = "% Generated by scripts/paper_assets.py from committed run outputs. Do not edit.\n"
        path.write_text(header + "\n".join(self.lines) + "\n", encoding="utf-8")


def _write(name: str, rows: list[str]) -> None:
    (OUT / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


class Protocol:
    """One evaluated regime of the run: its metrics, budget summaries and tables."""

    def __init__(self, tag: str) -> None:
        metrics_path, tables = PROTOCOLS[tag]
        self.tag = tag
        self.models = _by_model(_load(metrics_path))
        self.budgets = _load(tables / "budget_summary.json")
        with (tables / "day_sizes.csv").open(encoding="utf-8") as handle:
            self.day_rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        self.days = [(row["events"], row["positives"]) for row in self.day_rows]
        self.pairs = _load(tables / "pairwise_comparisons.json")
        self.tables = tables
        first = next(iter(self.budgets.values()))
        self.campaigns = int(first["n_campaigns"])
        self.accounts = int(first["n_accounts"])

    def first_detection(self, model: str) -> int:
        value = self.budgets[model]["statistics"]["first_detection"]["value"]
        operational = self.models[model]["operational"]["budget_for_first_detection"]
        if int(value) != int(operational):
            raise ValueError(f"{self.tag} {model}: first detection {value} != {operational}")
        return int(value)

    def null_p(self, model: str) -> float:
        """P(random first detection <= b*), exact; the run's simulated value is checked against it."""
        stored = float(
            self.budgets[model]["statistics"]["first_detection"]["random_ranking_at_least_as_good"]
        )
        exact = 1 - random_first_detection_survival(self.days, self.first_detection(model))
        # The ledger simulates the null with ledger.n_simulations (10,000) rankings.
        if abs(stored - exact) > 4 * math.sqrt(exact * (1 - exact) / NULL_SIMULATIONS) + 1e-4:
            raise ValueError(f"{self.tag} {model}: null {stored} != exact {exact}")
        return exact

    def metric(self, model: str, key: str) -> float:
        row = self.models[model]
        if key == "auc_pr":
            return float(row["auc_pr"])
        if key == "recall_fpr":
            return float(row["literature_comparable"]["recall_at_fixed_fpr"][FPR_KEY])
        return float(row["literature_comparable"][key])

    def recall(self, model: str, budget: int, unit: str = "event") -> float:
        operational = self.models[model]["operational"]
        index = operational["budgets"].index(budget)
        if unit == "event":
            return float(operational["campaign_recall"][index])
        return float(operational["alert_units"][unit]["campaign_recall"][index])

    def non_trivial(self) -> list[str]:
        """Non-floor detectors; M1 dropped where it is M1u (no validation, uniform weights)."""
        names = [m for m in ORDER if m in self.models and m not in FLOORS]
        if self.tag == "Harm":
            names.remove("M1_rules")
        return names


def _check_harmonised_m1(harm: Protocol) -> None:
    a, b = harm.models["M1_rules"], harm.models["M1u_rules_uniform"]
    if a["auc_pr"] != b["auc_pr"] or harm.first_detection("M1_rules") != harm.first_detection(
        "M1u_rules_uniform"
    ):
        raise ValueError("Harmonised M1 and M1u differ; the text says they are identical.")


def _model_macros(macros: Macros, p: Protocol) -> None:
    for model in p.models:
        prefix = p.tag + MACRO_NAMES[model]
        macros.add(f"{prefix}RocAuc", f"{p.metric(model, 'roc_auc'):.3f}")
        macros.add(f"{prefix}AucPr", _sci(p.metric(model, "auc_pr")))
        macros.add(f"{prefix}RecallFpr", _pct(p.metric(model, "recall_fpr")))
        macros.add(f"{prefix}FirstDetection", _int(p.first_detection(model)))
        macros.add(f"{prefix}NullP", _p(p.null_p(model)))
        for budget, name in BUDGET_NAMES.items():
            macros.add(f"{prefix}Recall{name}", _pct(p.recall(model, budget)))
            macros.add(f"{prefix}UserDay{name}", _pct(p.recall(model, budget, "user_day")))
            macros.add(f"{prefix}EdgeHour{name}", _pct(p.recall(model, budget, "edge_hour")))
        units = p.models[model]["operational"]["alert_units"]
        macros.add(f"{prefix}UserDayFirst", _int(units["user_day"]["budget_for_first_detection"]))
        macros.add(f"{prefix}EdgeHourFirst", _int(units["edge_hour"]["budget_for_first_detection"]))
        fixed = p.models[model]["operational"]["fixed_threshold"][0]
        macros.add(f"{prefix}FixedTenMean", _int(fixed["alerts_per_day_mean"]))
        macros.add(f"{prefix}FixedTenMax", _int(fixed["alerts_per_day_max"]))


def _protocol_macros(macros: Macros, p: Protocol) -> None:
    t = p.tag
    events = sum(n for n, _ in p.days)
    positives = sum(k for _, k in p.days)
    macros.add(f"{t}TestEvents", _int(events))
    macros.add(f"{t}TestPositives", _int(positives))
    macros.add(f"{t}TestRate", _sci(positives / events))
    macros.add(f"{t}TestCampaigns", _int(p.campaigns))
    macros.add(f"{t}TestAccounts", _int(p.accounts))
    macros.add(f"{t}TestDays", str(len(p.days)))
    for q, name in ((0.025, "Low"), (0.5, "Median"), (0.975, "High")):
        macros.add(f"{t}NullFirstDetection{name}", _int(random_first_detection_quantile(p.days, q)))
    macros.add(f"{t}NullWithinFiveHundred", _pct(1 - random_first_detection_survival(p.days, 500)))
    macros.add(f"{t}PairsTotal", str(len(p.pairs)))
    macros.add(f"{t}PairsSignificant", str(sum(c["significant"] for c in p.pairs)))
    macros.add(f"{t}NonTrivial", str(len(p.non_trivial())))
    macros.add(f"{t}UserDaysMin", _int(min(r["user_days"] for r in p.day_rows)))
    macros.add(f"{t}UserDaysMax", _int(max(r["user_days"] for r in p.day_rows)))
    macros.add(f"{t}EdgeHoursMin", _int(min(r["edge_hours"] for r in p.day_rows)))
    macros.add(f"{t}EdgeHoursMax", _int(max(r["edge_hours"] for r in p.day_rows)))
    if t == "TempTwo":
        return
    users = _load(p.tables / "bootstrap_user_clusters.json")
    macros.add(f"{t}UserClusterSignificant", str(users["n_significant"]))
    macros.add(f"{t}UserClusterUsers", str(users["n_users_with_positives"]))
    gaps = {
        int(g["gap_hours"]): int(g["n_campaigns"])
        for g in _load(p.tables / "campaign_gap_sensitivity.json")
    }
    for hours, name in ((1, "One"), (6, "Six"), (24, "TwentyFour"), (72, "SeventyTwo")):
        macros.add(f"{t}GapCampaigns{name}", str(gaps[hours]))


def _permutation_p(x: list[float], y: list[float], observed: float) -> float:
    """Two-sided Monte Carlo permutation p of Spearman's rho, seeded."""
    rx, ry = rankdata(x), rankdata(y)
    rng = np.random.default_rng(0)
    perms = np.argsort(rng.random((PERMUTATIONS, len(ry))), axis=1)
    shuffled = ry[perms]
    a = (rx - rx.mean()) / np.sqrt(((rx - rx.mean()) ** 2).sum())
    b = shuffled - shuffled.mean(axis=1, keepdims=True)
    b /= np.sqrt((b**2).sum(axis=1, keepdims=True))
    rhos = b @ a
    return float((1 + np.sum(np.abs(rhos) >= abs(observed) - 1e-12)) / (PERMUTATIONS + 1))


def _spearman(x: list[float], y: list[float]) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _correlations(macros: Macros, harm: Protocol, temp: Protocol) -> None:
    rows = []
    for key, (macro, label) in METRICS.items():
        cells = [label]
        for p in (harm, temp):
            names = p.non_trivial()
            metric = [p.metric(m, key) for m in names]
            budget = [-p.first_detection(m) for m in names]
            rho = _spearman(metric, budget)
            pvalue = _permutation_p(metric, budget, rho)
            macros.add(f"{p.tag}Spearman{macro}", f"{rho:.2f}")
            macros.add(f"{p.tag}SpearmanP{macro}", _p(pvalue))
            cells += [f"{rho:.2f}", _p(pvalue)]
        rows.append(" & ".join(cells) + r" \\")
    _write("table_correlations.tex", rows)
    macros.add("PermutationCount", _int(PERMUTATIONS))


def _results_table(name: str, p: Protocol) -> None:
    rows = []
    for model in ORDER:
        if model not in p.models:
            continue
        row = p.models[model]
        label = LABELS[model] + (" (floor)" if model in FLOORS else "")
        cells = [
            label,
            f"${_sci(row['auc_pr'])}$",
            f"[${_sci(row['auc_pr_ci_low'])}$, ${_sci(row['auc_pr_ci_high'])}$]",
            f"{p.metric(model, 'roc_auc'):.3f}",
            _pct(p.metric(model, "recall_fpr")),
            _pct(p.recall(model, 50)),
            _pct(p.recall(model, 500)),
            _int(p.first_detection(model)),
            _p(p.null_p(model)),
        ]
        rows.append(" & ".join(cells) + r" \\")
    _write(name, rows)


def _regime_table(macros: Macros, r1: Protocol, r2: Protocol) -> None:
    """The temporal detectors whose AUC-PR or first detection moves by more than 1% in R2."""
    rows, unchanged = [], []
    for model in ORDER:
        if model not in r1.models:
            continue
        a1, a2 = r1.metric(model, "auc_pr"), r2.metric(model, "auc_pr")
        b1, b2 = r1.first_detection(model), r2.first_detection(model)
        if abs(a2 - a1) / a1 > 0.01 or abs(b2 - b1) / b1 > 0.01:
            rows.append(
                " & ".join(
                    [
                        LABELS[model],
                        f"${_sci(a1)}$",
                        f"${_sci(a2)}$",
                        f"{r1.metric(model, 'roc_auc'):.3f}",
                        f"{r2.metric(model, 'roc_auc'):.3f}",
                        _int(b1),
                        _int(b2),
                    ]
                )
                + r" \\"
            )
        else:
            unchanged.append(model)
    _write("table_regimes.tex", rows)
    macros.add("RegimeUnchanged", str(len(unchanged)))


def _units_table(p: Protocol) -> None:
    rows = []
    for model in sorted(p.non_trivial() + ["M0a_random"], key=p.first_detection):
        units = p.models[model]["operational"]["alert_units"]
        fixed = p.models[model]["operational"]["fixed_threshold"][0]
        cells = [
            LABELS[model],
            _pct(p.recall(model, 500)),
            _pct(p.recall(model, 500, "edge_hour")),
            _pct(p.recall(model, 50, "user_day")),
            _pct(p.recall(model, 500, "user_day")),
            _int(units["user_day"]["budget_for_first_detection"]),
            _int(fixed["alerts_per_day_mean"]),
            _int(fixed["alerts_per_day_max"]),
        ]
        rows.append(" & ".join(cells) + r" \\")
    _write(f"table_units_{p.tag.lower()}.tex", rows)


def _fit_variance(macros: Macros, harm: Protocol, temp: Protocol) -> None:
    rows = []
    for p in (harm, temp):
        variance = _load(p.tables / "fit_variance.json")
        for model in ORDER:
            if model not in variance:
                continue
            runs = variance[model]["runs"]
            aucs = [r["auc_pr"] for r in runs]
            firsts = [r["budget_for_first_detection"] for r in runs]
            at_500 = [r["campaign_recall"]["500"] for r in runs]
            prefix = p.tag + MACRO_NAMES[model]
            macros.add(f"{prefix}FitFirstMin", _int(min(firsts)))
            macros.add(f"{prefix}FitFirstMax", _int(max(firsts)))
            rows.append(
                " & ".join(
                    [
                        {"Harm": "harmonised", "Temp": "temporal"}[p.tag],
                        LABELS[model],
                        f"[${_sci(min(aucs))}$, ${_sci(max(aucs))}$]",
                        f"[{_int(min(firsts))}, {_int(max(firsts))}]",
                        f"[{_pct(min(at_500))}, {_pct(max(at_500))}]",
                    ]
                )
                + r" \\"
            )
        macros.add(f"{p.tag}FitSeeds", str(len(next(iter(variance.values()))["runs"])))
    _write("table_fit_variance.tex", rows)


def _split(macros: Macros, harm: Protocol) -> None:
    """Split table: temporal partitions from the September summary (same partitions, checked)."""
    splits = _load(WIDE / "split_summary.json")
    log = (RUN / "evaluate_temporal.log").read_text(encoding="utf-8")
    train_temporal = int(re.search(r"sample of \d+/(\d+) training rows", log).group(1))  # type: ignore[union-attr]
    if train_temporal != splits["train"]["events"]:
        raise ValueError("Temporal training partition differs from the September summary.")
    temp_days = Protocol("Temp").days
    if [list(d) for d in temp_days] != [
        [d["events"], d["positives"]] for d in splits["test"]["days"]
    ]:
        raise ValueError("Temporal test days differ from the September summary.")
    log = (RUN / "evaluate_harmonised.log").read_text(encoding="utf-8")
    train_harm = int(re.search(r"sample of \d+/(\d+) training rows", log).group(1))  # type: ignore[union-attr]
    macros.add("HarmTrainEvents", _int(train_harm))
    macros.add("TempTrainEvents", _int(splits["train"]["events"]))
    macros.add("TempValPositives", _int(splits["val"]["positives"]))
    macros.add("TempTrainPositives", _int(splits["train"]["positives"]))
    names = {"train": "Train", "val": "Validation", "test": "Test"}
    out = []
    for i, (name, split) in enumerate(splits.items()):
        out.append(
            (r"\multirow{3}{*}{temporal} & " if i == 0 else " & ")
            + " & ".join(
                [
                    names[name],
                    f"days {split['first_day']}--{split['last_day']}",
                    _int(split["events"]),
                    _int(split["positives"]),
                    _int(split["campaigns"]),
                    f"${_sci(split['positive_rate'])}$",
                ]
            )
            + r" \\"
        )
    harm_events = sum(n for n, _ in harm.days)
    harm_positives = sum(k for _, k in harm.days)
    out.append(r"\midrule")
    out.append(
        r"\multirow{2}{*}{harmonised} & Train & hours 0--41 & "
        + f"{_int(train_harm)} & 0 & 0 & 0 \\\\"
    )
    out.append(
        " & Test & hours 41--1392 & "
        + " & ".join(
            [
                _int(harm_events),
                _int(harm_positives),
                _int(harm.campaigns),
                f"${_sci(harm_positives / harm_events)}$",
            ]
        )
        + r" \\"
    )
    _write("table_split.tex", out)
    for day in splits["test"]["days"]:
        name = {11: "Eleven", 12: "Twelve", 13: "Thirteen"}[day["day"]]
        macros.add(f"TempTestDay{name}Positives", _int(day["positives"]))
    macros.add("TempTestTopHostShare", _pct(splits["test"]["top_source_host_share"]))
    macros.add("TempTestTopHostCampaigns", _int(splits["test"]["source_hosts"][0]["campaigns"]))


def _concentration(macros: Macros) -> None:
    data = _load(RUN / "diagnostics" / "redteam_concentration.json")
    top = data["all"]["top_sources"][0]
    macros.add("TopHost", top["src_computer"])
    macros.add("RedTeamRows", _int(data["distinct_rows"]))
    macros.add("RedTeamTopHostEvents", _int(top["events"]))
    macros.add("RedTeamTopHostShare", _pct(top["share"]))
    macros.add("RedTeamSourceComputers", str(data["all"]["source_computers"]))
    macros.add("RedTeamAccounts", _int(data["all"]["accounts"]))
    if data["harmonised_test"]["events"] != data["distinct_rows"]:
        raise ValueError("The text says every red-team event falls in the harmonised test window.")


def _rules(macros: Macros) -> None:
    pattern = re.compile(r"^(R\d_\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$")
    tables: dict[str, dict[str, float]] = {}
    for tag, protocol in (("Temp", "temporal"), ("Harm", "harmonised")):
        text = (RUN / "diagnostics" / f"m1u_rules_{protocol}.txt").read_text(encoding="utf-8")
        tables[tag] = {
            m.group(1): float(m.group(4))
            for line in text.splitlines()
            if (m := pattern.match(line.strip()))
        }
        sample = re.search(r"sample: (\d+) malicious, (\d+) benign", text)
        machine = re.search(r"machine accounts: ([\d.]+) of benign, ([\d.]+) of malicious", text)
        assert sample and machine
        macros.add(f"{tag}RuleSampleMalicious", _int(int(sample.group(1))))
        macros.add(f"{tag}RuleSampleBenign", _int(int(sample.group(2))))
        macros.add(f"{tag}MachineBenignShare", _pct(float(machine.group(1))))
        macros.add(f"{tag}MachineMaliciousShare", _pct(float(machine.group(2))))
    rows = [
        f"{label} & {tables['Temp'][rule]:.2f} & {tables['Harm'][rule]:.2f} \\\\"
        for rule, label in RULES
    ]
    _write("table_rules_lanl.tex", rows)
    below = {tag: sum(v < 0.5 for v in values.values()) for tag, values in tables.items()}
    macros.add("TempRulesBelowChance", str(below["Temp"]))
    macros.add("HarmRulesBelowChance", str(below["Harm"]))


def _euler(macros: Macros) -> None:
    rows = []
    for config, label in EULER_CONFIGS:
        data = _load(RUN / f"euler_fidelity_{config}.json")
        name = "EulerFid" + "".join(w.title() for w in config.split("_"))
        macros.add(f"{name}Auc", f"{data['auc_mean']:.4f}")
        macros.add(f"{name}AucSd", f"{data['auc_std']:.4f}")
        macros.add(f"{name}Ap", f"{data['ap_mean']:.5f}")
        macros.add(f"{name}ApSd", f"{data['ap_std']:.5f}")
        macros.add(f"{name}Seeds", str(len(data["runs"])))
        rows.append(
            " & ".join(
                [
                    label,
                    _int(data["events_kept"]),
                    f"{_int(data['test_edges'])} ({data['anomalous_edges']})",
                    f"{data['auc_mean']:.4f} $\\pm$ {data['auc_std']:.4f}",
                    f"{data['ap_mean']:.5f} $\\pm$ {data['ap_std']:.5f}",
                ]
            )
            + r" \\"
        )
        if config == "ntlm":
            macros.add("EulerTestEdges", _int(data["test_edges"]))
            macros.add("EulerAnomalousEdges", str(data["anomalous_edges"]))
            macros.add("EulerNtlmEvents", _int(data["events_kept"]))
    # The authors' code: results.txt holds the mean and the standard error over its repetitions.
    lines = (RUN / "euler_original" / "results.txt").read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("\tTPR")).split("\t")
    values = {
        line.split("\t")[0]: dict(zip(header, line.split("\t"), strict=False))
        for line in lines
        if line.split("\t")[0] in ("mean", "stderr")
    }
    runs = [line.split(", ") for line in lines if line.startswith("PredictorRecurrent, ")]
    n = len(runs)
    auc, ap = float(values["mean"]["AUC"]), float(values["mean"]["AP"])
    auc_sd = float(values["stderr"]["AUC"]) * math.sqrt(n)
    ap_sd = float(values["stderr"]["AP"]) * math.sqrt(n)
    scored = {round(float(r[3]) / float(r[1])) for r in runs}
    if len(scored) != 1:
        raise ValueError(f"TP/TPR differs across the authors' runs: {scored}")
    loaded = (RUN / "euler_original" / "authors_test_load.out").read_text(encoding="utf-8")
    total = re.search(r"total: (\d+) snapshots, (\d+) edges, (\d+) anomalous", loaded)
    first = re.search(r"never scored.*?: (\d+)", loaded)
    assert total and first
    evaluated = int(total.group(3)) - int(first.group(1))
    if {evaluated} != scored:
        raise ValueError("Loaded anomalous edges minus the unscored first snapshot != TP/TPR.")
    macros.add("EulerAuthorsRuns", str(n))
    macros.add("EulerAuthorsAuc", f"{auc:.4f}")
    macros.add("EulerAuthorsAucSd", f"{auc_sd:.4f}")
    macros.add("EulerAuthorsAp", f"{ap:.4f}")
    macros.add("EulerAuthorsApSd", f"{ap_sd:.4f}")
    macros.add("EulerAuthorsApMin", f"{min(float(r[7]) for r in runs):.4f}")
    macros.add("EulerAuthorsApMax", f"{max(float(r[7]) for r in runs):.4f}")
    macros.add("EulerAuthorsLoadedEdges", _int(int(total.group(2))))
    macros.add("EulerAuthorsLoadedAnomalous", total.group(3))
    macros.add("EulerAuthorsFirstSnapshot", first.group(1))
    macros.add("EulerAuthorsEvaluated", str(evaluated))
    rows.append(
        " & ".join(
            [
                "Authors' code, this machine",
                "NTLM, as above",
                f"{_int(int(total.group(2)))} ({evaluated} scored)",
                f"{auc:.4f} $\\pm$ {auc_sd:.4f}",
                f"{ap:.5f} $\\pm$ {ap_sd:.5f}",
            ]
        )
        + r" \\"
    )
    rows.append(
        " & ".join(
            [
                r"Published \parencite{king2022euler}",
                _int(EULER_PUBLISHED["events"]),
                f"({EULER_PUBLISHED['anomalous']})",
                f"{EULER_PUBLISHED['auc']:.4f}",
                f"{EULER_PUBLISHED['ap']:.4f}",
            ]
        )
        + r" \\"
    )
    _write("table_euler.tex", rows)
    macros.add("EulerPublishedAuc", f"{EULER_PUBLISHED['auc']:.4f}")
    macros.add("EulerPublishedAp", f"{EULER_PUBLISHED['ap']:.4f}")
    macros.add("EulerPublishedAnomalous", str(EULER_PUBLISHED["anomalous"]))

    edges = (RUN / "euler_original" / "authors_test_edges.out").read_text(encoding="utf-8")
    theirs = re.search(
        r"authors alignment: (\d+) edges \((\d+) in the last snapshot.*?anomalous (\d+).*?labelled (\d+)",
        edges,
    )
    ours = re.search(r"AuthBench alignment: (\d+) edges, anomalous (\d+).*?labelled (\d+)", edges)
    matches = (RUN / "euler_original" / "redteam_matches.out").read_text(encoding="utf-8")
    matched = re.search(
        r"red-team rows \(distinct\): (\d+)\s+with any matching event: (\d+)\s+with an NTLM one: (\d+)",
        matches,
    )
    assert theirs and ours and matched
    if int(theirs.group(4)) != EULER_PUBLISHED["anomalous"]:
        raise ValueError("The authors' alignment no longer gives the published anomalous count.")
    if int(ours.group(1)) != _load(RUN / "euler_fidelity_ntlm.json")["test_edges"]:
        raise ValueError("The authors' files in AuthBench's alignment differ from AuthBench's set.")
    macros.add("EulerTheirAlignmentEdges", _int(int(theirs.group(1))))
    macros.add("EulerTheirLastSnapshot", _int(int(theirs.group(2))))
    macros.add("EulerTheirAnomalous", theirs.group(3))
    macros.add("EulerOurAlignmentAnomalous", ours.group(2))
    macros.add("RedTeamMatched", matched.group(2))
    macros.add("RedTeamUnmatched", str(int(matched.group(1)) - int(matched.group(2))))


def _staffing(macros: Macros) -> None:
    staffing = {
        row["budget"]: row for row in _load(RUN / "harmonised" / "tables" / "staffing.json")
    }
    for budget, name in BUDGET_NAMES.items():
        low, high = staffing[budget]["analysts_at_10_to_20_min"]
        macros.add(f"Staff{name}Low", f"{low:.1f}")
        macros.add(f"Staff{name}High", f"{high:.1f}")
        macros.add(f"Staff{name}Share", _pct(staffing[budget]["share_of_soc_daily_capacity"], 0))


def _earlier(macros: Macros, temp: Protocol) -> None:
    wide = _by_model(_load(WIDE / "metrics_summary.json"))
    rows = []
    for model in ORDER:
        if model not in wide:
            continue
        old, new = wide[model], temp.models[model]
        rows.append(
            " & ".join(
                [
                    LABELS[model],
                    f"${_sci(old['auc_pr'])}$",
                    f"${_sci(new['auc_pr'])}$",
                    f"{old['literature_comparable']['roc_auc']:.3f}",
                    f"{new['literature_comparable']['roc_auc']:.3f}",
                    _int(old["operational"]["budget_for_first_detection"]),
                    _int(temp.first_detection(model)),
                ]
            )
            + r" \\"
        )
    _write("table_earlier.tex", rows)
    macros.add("WideModels", str(len(rows)))
    macros.add(
        "WidePcaFirstDetection",
        _int(wide["M2b_pca_reconstruction"]["operational"]["budget_for_first_detection"]),
    )
    run = _load(RUN / "run.json")
    macros.add("RunHours", f"{run['hours']:.1f}")


def _demo_rules(macros: Macros) -> None:
    with (DEMO / "rule_discrimination.csv").open(encoding="utf-8") as handle:
        rules = list(csv.DictReader(handle))
    rows = []
    for r in rules:
        ratio = r["separation_ratio"]
        rows.append(
            " & ".join(
                [
                    r["rule"].replace("_", r"\_"),
                    f"{float(r['mean_score_benign']):.4f}",
                    f"{float(r['mean_score_malicious']):.4f}",
                    f"{float(ratio):.1f}" if ratio else "n/a",
                    f"{float(r['calibrated_weight']):.3f}",
                ]
            )
            + r" \\"
        )
    _write("table_rules_demo.tex", rows)
    seven = {r["rule"]: r for r in rules}["R7_lateral_chain"]
    macros.add("DemoRSevenRatio", f"{float(seven['separation_ratio']):.0f}")
    macros.add("DemoRSevenWeight", f"{float(seven['calibrated_weight']):.3f}")


def _tie_macros(macros: Macros, p: Protocol) -> None:
    operational = p.models["M2a_pair_rarity"]["operational"]
    brackets = operational["campaign_recall_tie_bracket"]
    macros.add(f"{p.tag}PairRarityBracketTen", _pct(brackets[0][1]))
    macros.add(f"{p.tag}PairRarityBracketFiveHundred", _pct(brackets[-1][1]))
    macros.add(
        f"{p.tag}PairRarityContenders", _int(operational["tie_exposure"][-1]["n_contenders"])
    )
    others = [
        m
        for m in p.models
        if m not in ("M2a_pair_rarity", "M0b_always_fail")
        and any(
            low != high for low, high in p.models[m]["operational"]["campaign_recall_tie_bracket"]
        )
    ]
    if others:
        raise ValueError(f"{p.tag}: tie brackets of non-zero width beyond M2a and M0b: {others}")


def _figure_first_detection(path: Path, harm: Protocol, temp: Protocol) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4))
    for row, (p, title) in enumerate(((harm, "harmonised"), (temp, "temporal, R1"))):
        low, median, high = (
            random_first_detection_quantile(p.days, q) for q in (0.025, 0.5, 0.975)
        )
        for col, (key, xlabel) in enumerate(
            (("roc_auc", "ROC-AUC"), ("auc_pr", "AUC-PR (log scale)"))
        ):
            ax = axes[row][col]
            ax.axhspan(1, 500, color="0.93", zorder=0)
            ax.axhspan(low, high, color="0.80", zorder=0.5)
            ax.axhline(median, color="0.35", linestyle="--", linewidth=0.8, zorder=1)
            for model in p.non_trivial():
                x, y = p.metric(model, key), p.first_detection(model)
                if key == "auc_pr":
                    r = p.models[model]
                    ax.errorbar(
                        x,
                        y,
                        xerr=[[x - r["auc_pr_ci_low"]], [r["auc_pr_ci_high"] - x]],
                        fmt="none",
                        ecolor="0.3",
                        elinewidth=0.5,
                        capsize=1.5,
                        zorder=2,
                    )
                ax.scatter(x, y, s=12, color="black", zorder=3)
                dx, dy = LABEL_OFFSETS[(p.tag, key)].get(model, (3, 2))
                far = abs(dx) > 8 or abs(dy) > 8
                ax.annotate(
                    CODES[model],
                    (x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=6.5,
                    ha="center" if far else "left",
                    va="center" if far else "bottom",
                    arrowprops={"arrowstyle": "-", "color": "0.4", "linewidth": 0.4}
                    if far
                    else None,
                )
            ax.set_yscale("log")
            ax.set_ylim(1, 10**6)
            ax.grid(True, which="major", linewidth=0.3, color="0.8")
            if key == "auc_pr":
                ax.set_xscale("log")
                ax.set_xlim(1e-6, 1e-2)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                ax.set_xlim(0.2, 1.08)
                ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(f"{title}\nfirst detection (alerts/day)")
            ax.text(
                0.98, 0.03, "1 to 500 alerts/day", transform=ax.transAxes, fontsize=6, ha="right"
            )
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    plt.close(fig)


def _campaign_budgets(p: Protocol) -> dict[str, list[int]]:
    """b(c) per campaign and model: the best daily rank of the campaign's events, arrival tie order."""
    best: dict[str, dict[int, int]] = defaultdict(dict)
    with (p.tables / "positives_ledger.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            campaign, rank = int(row["campaign_id"]), int(row["rank"])
            current = best[row["model"]].get(campaign)
            best[row["model"]][campaign] = rank if current is None else min(current, rank)
    out = {}
    for model, by_campaign in best.items():
        if len(by_campaign) != p.campaigns:
            raise ValueError(f"{p.tag} {model}: {len(by_campaign)} campaigns in the ledger")
        out[model] = sorted(by_campaign.values())
        if out[model][0] != p.first_detection(model):
            raise ValueError(f"{p.tag} {model}: ledger first detection differs")
    return out


def _figure_recall(path: Path, harm: Protocol, temp: Protocol) -> None:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    grid = np.unique(np.round(np.logspace(0, 6, 400)).astype(int))
    styles = ["-", "--", "-.", ":"]
    for ax, (p, title) in zip(axes, ((harm, "harmonised"), (temp, "temporal, R1")), strict=True):
        budgets = _campaign_budgets(p)
        ax.axvspan(10, 500, color="0.93", zorder=0)
        for model in p.non_trivial():
            if model in CURVE_HIGHLIGHT[p.tag]:
                continue
            b = np.array(budgets[model])
            ax.step(
                grid,
                (b[None, :] <= grid[:, None]).mean(axis=1),
                where="post",
                color="0.75",
                linewidth=0.6,
            )
        for model, style in zip(CURVE_HIGHLIGHT[p.tag], styles, strict=True):
            b = np.array(budgets[model])
            ax.step(
                grid,
                (b[None, :] <= grid[:, None]).mean(axis=1),
                where="post",
                color="black",
                linestyle=style,
                linewidth=0.9,
                label=LABELS[model],
            )
        b = np.array(budgets["M0a_random"])
        ax.step(
            grid,
            (b[None, :] <= grid[:, None]).mean(axis=1),
            where="post",
            color="0.45",
            linewidth=1.4,
            linestyle=(0, (4, 1, 1, 1, 1, 1)),
            label="M0a random (floor)",
        )
        ax.set_xscale("log")
        ax.set_xlim(1, 10**6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Daily budget (alerts/day)")
        ax.set_title(title, fontsize=8)
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
        ax.legend(fontsize=6, loc="upper left", frameon=False)
    axes[0].set_ylabel("Campaign recall")
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None, "Producer": None})
    # The README shows the same figure; GitHub does not render PDF inline.
    fig.savefig(path.with_suffix(".png"), dpi=200, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("table_*.tex"):
        stale.unlink()
    macros = Macros()
    harm, temp, temp_two = Protocol("Harm"), Protocol("Temp"), Protocol("TempTwo")
    _check_harmonised_m1(harm)
    for p in (harm, temp, temp_two):
        _model_macros(macros, p)
        _protocol_macros(macros, p)
    for p in (harm, temp):
        _tie_macros(macros, p)
    _correlations(macros, harm, temp)
    _results_table("table_results_harmonised.tex", harm)
    _results_table("table_results_temporal.tex", temp)
    _regime_table(macros, temp, temp_two)
    _units_table(harm)
    _units_table(temp)
    _fit_variance(macros, harm, temp)
    _split(macros, harm)
    _concentration(macros)
    _rules(macros)
    _euler(macros)
    _staffing(macros)
    _earlier(macros, temp)
    _demo_rules(macros)
    macros.write(OUT / "numbers.tex")
    _figure_first_detection(OUT / "figure_first_detection.pdf", harm, temp)
    _figure_recall(OUT / "figure_campaign_recall.pdf", harm, temp)


if __name__ == "__main__":
    main()
