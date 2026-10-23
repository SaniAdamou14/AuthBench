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
import gzip
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
from scipy.stats import t as student_t  # noqa: E402

from authbench.evaluate.budget import (  # noqa: E402
    random_first_detection_quantile,
    random_first_detection_survival,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "reports" / "runs" / "20261022T102638Z"
WIDE = ROOT / "reports" / "lanl" / "wider_window_results"
DEMO = ROOT / "reports" / "demo" / "tables"
OUT = ROOT / "reports" / "paper" / "generated"
# User-day activity recomputed from the log by scripts/user_day_activity.py, after the run.
USER_DAYS = RUN / "diagnostics" / "user_days"

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
UNIT_SIMULATIONS = 2_000
# Campaign-budget summaries from budget_summary.json: macro suffix and table label.
SUMMARIES = {
    "first_detection": ("First", "first detection $b^{*}$"),
    "budget_q10": ("Qten", "10\\% quantile of $b(c)$"),
    "budget_q25": ("Qtwentyfive", "25\\% quantile"),
    "budget_q50": ("Median", "median $b(c)$"),
    "log_area": ("LogArea", "log-area"),
}
# Label-free rankings of user-days by their activity that day: column, macro name, table label.
BASELINES = {
    "distinct_dst": ("Dst", "Most distinct destinations"),
    "distinct_src": ("Src", "Most distinct sources"),
    "events": ("Events", "Most events"),
    "failures": ("Failures", "Most failed authentications"),
}
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
EULER_PUBLISHED = {
    "events": 45_871_390,
    "nodes": 17_685,
    "anomalous": 518,
    "auc": 0.9906,
    "ap": 0.0155,
    "tpr": 0.8549,
    "fpr": 0.006088,
    "precision": 0.0050,
}
# Larroche (2026), Table 4 (LANL universe) and Table 5 (event-level re-evaluation, mean and
# standard deviation over ten runs, in percent; "rep." is the originally reported value).
LARROCHE_EVENTS = 369_569_626
LARROCHE_MALICIOUS = 702
LARROCHE_TABLE5 = {
    "Euler (link detection)": ("97.97 $\\pm$ 0.17", "99.12", "0.02 $\\pm$ 0.00", "5.23"),
    "Argus": ("98.44 $\\pm$ 0.78", "99.83", "0.09 $\\pm$ 0.04", "32.27"),
    "Pikachu": ("78.28 $\\pm$ 0.43", "94 $\\pm$ 0.4", "0.00 $\\pm$ 0.00", "n/a"),
}
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
        self.days_by_number = {row["day"]: row for row in self.day_rows}
        self.pairs = _load(tables / "pairwise_comparisons.json")
        self.tables = tables
        first = next(iter(self.budgets.values()))
        self.campaigns = int(first["n_campaigns"])
        self.accounts = int(first["n_accounts"])
        # Every positive event, with its campaign, day and daily rank (arrival tie order), per model.
        self.ledger: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        self.event_campaign: dict[int, int] = {}
        with (tables / "positives_ledger.csv").open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.ledger[row["model"]].append(
                    (int(row["campaign_id"]), int(row["day"]), int(row["rank"]))
                )
                if row["model"] == "M0a_random":
                    self.event_campaign[int(row["event_id"])] = int(row["campaign_id"])

    def campaign_budgets(self, model: str, skip_days: tuple[int, ...] = ()) -> list[int]:
        """Sorted b(c): each campaign's best daily rank, ignoring the days in `skip_days`."""
        best: dict[int, int] = {}
        for campaign, day, rank in self.ledger[model]:
            if day not in skip_days:
                best[campaign] = min(rank, best.get(campaign, rank))
        return sorted(best.values())

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
            macros.add(
                f"{prefix}Recall{name}Count", str(round(p.recall(model, budget) * p.campaigns))
            )
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


def _holm(pvalues: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _summary(p: Protocol, model: str, key: str) -> float:
    """A campaign-budget summary oriented so that higher is better."""
    if key == "first_detection":
        return -float(p.first_detection(model))
    value = float(p.budgets[model]["statistics"][key]["value"])
    return value if key == "log_area" else -value


def _correlations(macros: Macros, harm: Protocol, temp: Protocol) -> None:
    """Spearman's rho between threshold-free metrics and every budget summary, as one family.

    Per protocol, the family is every (metric, summary) pair; p-values are Monte Carlo
    permutation p-values, Holm-adjusted within the family. Leave-one-detector-out gives the
    range of rho when any single detector is removed.
    """
    metrics = {
        "roc_auc": ("RocAuc", "ROC-AUC"),
        "auc_pr": ("AucPr", "AUC-PR"),
        "cw_ap": ("CwAp", "campaign-weighted AP"),
    }
    results: dict[str, dict[tuple[str, str], tuple[float, float, float, float]]] = {}
    for p in (harm, temp):
        names = p.non_trivial()
        raw: dict[tuple[str, str], tuple[float, float, float, float]] = {}
        for key in metrics:
            values = [
                float(p.models[m]["auc_pr_campaign_weighted"])
                if key == "cw_ap"
                else p.metric(m, key)
                for m in names
            ]
            for summary in SUMMARIES:
                budget = [_summary(p, m, summary) for m in names]
                rho = _spearman(values, budget)
                drops = [
                    _spearman(values[:i] + values[i + 1 :], budget[:i] + budget[i + 1 :])
                    for i in range(len(names))
                ]
                raw[(key, summary)] = (
                    rho,
                    _permutation_p(values, budget, rho),
                    min(drops),
                    max(drops),
                )
        holm = _holm([v[1] for v in raw.values()])
        results[p.tag] = {
            k: (v[0], v[1], h, v[2], v[3]) for (k, v), h in zip(raw.items(), holm, strict=True)
        }
        for (key, summary), (rho, pvalue, adjusted, low, high) in results[p.tag].items():
            name = f"{p.tag}Rho{metrics[key][0]}{SUMMARIES[summary][0]}"
            macros.add(name, f"{rho:.2f}")
            macros.add(f"{name}P", _p(pvalue))
            macros.add(f"{name}Holm", _p(adjusted))
            macros.add(f"{name}Low", f"{low:.2f}")
            macros.add(f"{name}High", f"{high:.2f}")
        zeros = sum(p.metric(m, "recall_fpr") == 0 for m in names)
        macros.add(f"{p.tag}RecallFprZeros", str(zeros))
        recall = [p.metric(m, "recall_fpr") for m in names]
        rho = _spearman(recall, [_summary(p, m, "first_detection") for m in names])
        macros.add(f"{p.tag}RhoRecallFprFirst", f"{rho:.2f}")
    main, full = [], []
    for summary, (_, label) in SUMMARIES.items():
        cells = [label]
        for tag in ("Harm", "Temp"):
            for key in metrics:
                rho, _, adjusted, _, _ = results[tag][(key, summary)]
                cell = f"{rho:.2f} ({_p(adjusted)})"
                cells.append(f"\\textbf{{{cell}}}" if adjusted < 0.05 else cell)
        main.append(" & ".join(cells) + r" \\")
    for tag, protocol in (("Harm", "harmonised"), ("Temp", "temporal")):
        for key, (_, label) in metrics.items():
            for summary, (_, slabel) in SUMMARIES.items():
                rho, pvalue, adjusted, low, high = results[tag][(key, summary)]
                full.append(
                    f"{protocol} & {label} & {slabel} & {rho:.2f} & [{low:.2f}, {high:.2f}] & "
                    f"{_p(pvalue)} & {_p(adjusted)} \\\\"
                )
    _write("table_correlations.tex", main)
    _write("table_correlations_full.tex", full)
    macros.add("PermutationCount", _int(PERMUTATIONS))
    macros.add("CorrelationFamily", str(len(metrics) * len(SUMMARIES)))


def _holm_null(p: Protocol) -> dict[str, float]:
    """Null probabilities of the first detection, Holm-adjusted over the non-trivial detectors."""
    names = p.non_trivial()
    return dict(zip(names, _holm([p.null_p(m) for m in names]), strict=True))


def _results_table(macros: Macros, name: str, p: Protocol) -> None:
    holm = _holm_null(p)
    for model, adjusted in holm.items():
        macros.add(f"{p.tag}{MACRO_NAMES[model]}NullHolm", _p(adjusted))
    rows, intervals = [], []
    for model in ORDER:
        if model not in p.models:
            continue
        row = p.models[model]
        label = LABELS[model] + (" (floor)" if model in FLOORS else "")
        cells = [
            label,
            f"{p.metric(model, 'roc_auc'):.3f}",
            f"${_sci(row['auc_pr'], 1)}$",
            f"${_sci(float(row['auc_pr_campaign_weighted']), 1)}$",
            _pct(p.metric(model, "recall_fpr")),
            str(round(p.recall(model, 50) * p.campaigns)),
            str(round(p.recall(model, 500) * p.campaigns)),
            _int(p.first_detection(model)),
            _p(p.null_p(model)),
            _p(holm[model]) if model in holm else "",
        ]
        rows.append(" & ".join(cells) + r" \\")
        intervals.append(
            f"{label} & ${_sci(row['auc_pr'])}$ & "
            f"[${_sci(row['auc_pr_ci_low'])}$, ${_sci(row['auc_pr_ci_high'])}$] \\\\"
        )
    _write(name, rows)
    _write(name.replace("results", "aucpr"), intervals)


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


class UserDays:
    """The user-days of a protocol's test days with red-team events, from `user_day_activity.py`.

    The harness keys a user-day by the source user without its domain, so one unit can hold
    events of two campaigns (one user name under two domains); a campaign is detected in user-days
    when one of its units is in the top k of its day. Every day is checked against the run:
    events, user-days and positives per day, and the positives' event identifiers.
    """

    def __init__(self, p: Protocol) -> None:
        name = {"Harm": "harmonised", "Temp": "temporal"}[p.tag]
        self.p = p
        self.summary = _load(USER_DAYS / "user_day_activity.json")[name]
        with gzip.open(USER_DAYS / f"user_days_{name}.csv.gz", "rt", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (USER_DAYS / f"positives_{name}.csv").open(encoding="utf-8") as handle:
            self.positives = list(csv.DictReader(handle))
        expected = {r["day"]: r for r in p.day_rows}
        self.days: dict[int, dict[str, np.ndarray]] = {}
        for day in sorted({int(r["day"]) for r in rows}):
            day_rows = sorted(
                (r for r in rows if int(r["day"]) == day),
                # The harness's unit tie order: first event time, then the unit key.
                key=lambda r: (int(r["first_time"]), r["src_user"]),
            )
            self.days[day] = {
                "user": np.array([r["src_user"] for r in day_rows]),
                **{
                    c: np.array([int(r[c]) for r in day_rows])
                    for c in ("events", "distinct_dst", "distinct_src", "failures")
                },
            }
        positive_days = {d for _, d, _ in p.ledger["M0a_random"]}
        if positive_days - set(self.days):
            raise ValueError(
                f"{p.tag}: positives on days {positive_days - set(self.days)} not read"
            )
        for day, units in self.days.items():
            row = expected[day]
            here = sum(int(x["day"]) == day for x in self.positives)
            if (int(units["events"].sum()), units["user"].size, here) != (
                row["events"],
                row["user_days"],
                row["positives"],
            ):
                raise ValueError(f"{p.tag} day {day}: user-day activity differs from day_sizes.csv")
        if {int(x["event_id"]) for x in self.positives} != set(p.event_campaign):
            raise ValueError(f"{p.tag}: the positives read are not the ledger's")
        # Per day, each positive unit's index and the campaigns it holds.
        self.units: dict[int, dict[int, set[int]]] = {}
        for x in self.positives:
            day = int(x["day"])
            index = int(np.flatnonzero(self.days[day]["user"] == x["src_user"])[0])
            self.units.setdefault(day, {}).setdefault(index, set()).add(
                p.event_campaign[int(x["event_id"])]
            )
        self.campaigns = sorted(set(p.event_campaign.values()))

    def pairs(self) -> int:
        return sum(len(c) for units in self.units.values() for c in units.values())

    def shared(self) -> int:
        return sum(len(c) > 1 for units in self.units.values() for c in units.values())

    def budgets(self, column: str) -> list[int]:
        """b(c) in user-days when each day's units are ranked by `column`, ties in the harness's order."""
        best: dict[int, int] = {}
        for day, units in self.units.items():
            values = self.days[day][column]
            # Stable sort on the tie-ordered rows: descending value, then the harness's tie order.
            order = np.argsort(-values, kind="stable")
            rank = np.empty(order.size, dtype=np.int64)
            rank[order] = np.arange(1, order.size + 1)
            for index, campaigns in units.items():
                for c in campaigns:
                    best[c] = min(int(rank[index]), best.get(c, int(rank[index])))
        return sorted(best[c] for c in self.campaigns)

    def null(self, size_aware: bool) -> tuple[dict[int, np.ndarray], np.ndarray]:
        """Campaign recall at 50 and 500 user-days, and b*, under a random ranking, per simulation.

        `size_aware=False` ranks the units uniformly at random. `size_aware=True` gives every event
        an independent uniform score and every unit the maximum of its events, as the harness
        aggregates a detector that ignores labels: a unit of n events then scores like U^(1/n),
        so its order is that of -log(U)/n, smaller first.
        """
        rng = np.random.default_rng(0 if size_aware else 1)
        index = {c: i for i, c in enumerate(self.campaigns)}
        detected = {k: np.zeros((UNIT_SIMULATIONS, len(index)), dtype=bool) for k in (50, 500)}
        first = np.full(UNIT_SIMULATIONS, np.iinfo(np.int64).max)
        for day, units in sorted(self.units.items()):
            n = self.days[day]["events"].astype(np.float64)
            positive = np.array(sorted(units))
            for start in range(0, UNIT_SIMULATIONS, 500):
                rows = slice(start, min(start + 500, UNIT_SIMULATIONS))
                key = rng.exponential(size=(rows.stop - rows.start, n.size))
                if size_aware:
                    key /= n
                at_positive = key[:, positive]
                best = at_positive.min(axis=1)
                first[rows] = np.minimum(first[rows], (key < best[:, None]).sum(axis=1) + 1)
                for k in (50, 500):
                    kth = np.partition(key, min(k, n.size) - 1, axis=1)[:, min(k, n.size) - 1]
                    inside = at_positive <= kth[:, None]
                    for j, unit in enumerate(positive):
                        for c in units[int(unit)]:
                            detected[k][rows, index[c]] |= inside[:, j]
        return {k: v.mean(axis=1) for k, v in detected.items()}, first


def _null_summary(share: np.ndarray) -> tuple[float, float, float]:
    return float(share.mean()), float(np.quantile(share, 0.025)), float(np.quantile(share, 0.975))


def _units_table(macros: Macros, p: Protocol) -> None:
    """User-day results with label-free activity rankings and two random-ranking nulls."""
    t = p.tag
    units = UserDays(p)
    counts = [
        (int(p.days_by_number[d]["user_days"]), len(units.units.get(d, {})))
        for d in sorted(p.days_by_number)
    ]
    macros.add(f"{t}PositiveUserDays", _int(sum(k for _, k in counts)))
    macros.add(f"{t}PositiveUserDayPairs", _int(units.pairs()))
    macros.add(f"{t}SharedPositiveUserDays", str(units.shared()))
    macros.add(f"{t}UnitNullFirstMedian", _int(random_first_detection_quantile(counts, 0.5)))
    uniform, _ = units.null(size_aware=False)
    sized, sized_first = units.null(size_aware=True)
    for name, recall in (("UnitNull", uniform), ("SizeNull", sized)):
        for budget, label in ((50, "Fifty"), (500, "FiveHundred")):
            mean, low, high = _null_summary(recall[budget])
            macros.add(f"{t}{name}{label}", _pct(mean))
            macros.add(f"{t}{name}{label}Low", _pct(low))
            macros.add(f"{t}{name}{label}High", _pct(high))
    macros.add(f"{t}SizeNullFirstMedian", _int(float(np.median(sized_first))))
    rows = []
    for model in sorted(p.non_trivial() + list(FLOORS), key=p.first_detection):
        first = int(
            p.models[model]["operational"]["alert_units"]["user_day"]["budget_for_first_detection"]
        )
        uniform_p = 1 - random_first_detection_survival(counts, first)
        sized_p = float(np.mean(sized_first <= first))
        macros.add(f"{t}{MACRO_NAMES[model]}UserDayNullP", _p(uniform_p))
        macros.add(f"{t}{MACRO_NAMES[model]}UserDaySizeNullP", _p(sized_p))
        rows.append(
            " & ".join(
                [
                    LABELS[model],
                    _pct(p.recall(model, 500)),
                    _pct(p.recall(model, 500, "edge_hour")),
                    _pct(p.recall(model, 50, "user_day")),
                    _pct(p.recall(model, 500, "user_day")),
                    _int(first),
                    _p(sized_p),
                ]
            )
            + r" \\"
        )
    rows.append(r"\midrule")
    best = 0.0
    for column, (name, label) in BASELINES.items():
        budgets = np.array(units.budgets(column))
        at = {k: float(np.mean(budgets <= k)) for k in (50, 500)}
        sized_p = float(np.mean(sized_first <= budgets[0]))
        best = max(best, at[500])
        macros.add(f"{t}Baseline{name}UserDayFifty", _pct(at[50]))
        macros.add(f"{t}Baseline{name}UserDayFiveHundred", _pct(at[500]))
        macros.add(f"{t}Baseline{name}UserDayFirst", _int(budgets[0]))
        macros.add(f"{t}Baseline{name}UserDayFiveHundredCount", str(int(np.sum(budgets <= 500))))
        rows.append(
            f"{label} & & & {_pct(at[50])} & {_pct(at[500])} & {_int(budgets[0])} & {_p(sized_p)} \\\\"
        )
    beaten = [m for m in p.non_trivial() if p.recall(m, 500, "user_day") > best]
    macros.add(f"{t}DetectorsAboveBaselines", str(len(beaten)))
    rows.append(r"\midrule")
    rows.append(
        "Random ranking of user-days & & & "
        f"{_pct(uniform[50].mean())} & {_pct(uniform[500].mean())} & "
        f"{_int(random_first_detection_quantile(counts, 0.5))} & \\\\"
    )
    rows.append(
        "Random event scores, maximum per user-day & & & "
        f"{_pct(sized[50].mean())} & {_pct(sized[500].mean())} & "
        f"{_int(float(np.median(sized_first)))} & \\\\"
    )
    _write(f"table_units_{t.lower()}.tex", rows)
    # Identity of the positives: red-team rows labelling two events, and campaigns off the top host.
    data = units.summary
    macros.add(f"{t}PositivesFromRows", _int(data["redteam_rows_matched"]))
    macros.add(f"{t}DoublyLabelled", str(data["positives"] - data["redteam_rows_matched"]))
    top = _load(RUN / "diagnostics" / "redteam_concentration.json")["all"]["top_sources"][0]
    off_host = {
        p.event_campaign[int(x["event_id"])]
        for x in units.positives
        if x["src_computer"] != top["src_computer"]
    }
    macros.add(f"{t}CampaignsOffTopHost", str(len(off_host)))
    # Campaigns are keyed by user@domain, the run's regrouping and bootstrap by user name alone.
    gaps = {
        int(g["gap_hours"]): int(g["n_campaigns"])
        for g in _load(p.tables / "campaign_gap_sensitivity.json")
    }
    by_name = _regroup(units.positives, ("src_user",), 24)
    by_account = _regroup(units.positives, ("src_user", "src_domain"), 24)
    if by_name != gaps[24]:
        raise ValueError(f"{t}: regrouping by user name gives {by_name}, the run {gaps[24]}")
    macros.add(f"{t}RegroupByAccount", str(by_account))
    macros.add(
        f"{t}TestAccountsWithDomain",
        str(len({(x["src_user"], x["src_domain"]) for x in units.positives})),
    )


def _regroup(positives: list[dict[str, str]], key: tuple[str, ...], gap_hours: int) -> int:
    """Campaigns when positives are grouped by `key` and split at gaps over `gap_hours`."""
    times: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for x in positives:
        times[tuple(x[k] for k in key)].append(int(x["time"]))
    return sum(
        1 + sum(b - a > gap_hours * 3600 for a, b in zip(ts, ts[1:], strict=False))
        for ts in (sorted(v) for v in times.values())
    )


def _fixed_table(p: Protocol) -> None:
    rows = []
    for model in sorted(p.non_trivial() + list(FLOORS), key=p.first_detection):
        cells = [LABELS[model]]
        for entry in p.models[model]["operational"]["fixed_threshold"]:
            if entry["budget"] in (10, 500):
                cells += [
                    _int(entry["alerts_per_day_mean"]),
                    _int(entry["alerts_per_day_max"]),
                    _pct(entry["campaign_recall"]),
                ]
        rows.append(" & ".join(cells) + r" \\")
    _write(f"table_fixed_{p.tag.lower()}.tex", rows)


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
            nulls = [1 - random_first_detection_survival(p.days, b) for b in firsts]
            at_500 = [r["campaign_recall"]["500"] for r in runs]
            reported = next(r for r in runs if r["seed"] == 42)["budget_for_first_detection"]
            if reported != p.first_detection(model):
                raise ValueError(f"{p.tag} {model}: seed 42 is not the reported fit")
            prefix = p.tag + MACRO_NAMES[model]
            macros.add(f"{prefix}FitFirstMin", _int(min(firsts)))
            macros.add(f"{prefix}FitFirstMax", _int(max(firsts)))
            macros.add(f"{prefix}FitFirstMedian", _int(float(np.median(firsts))))
            macros.add(f"{prefix}FitNullMedian", _p(float(np.median(nulls))))
            macros.add(f"{prefix}FitNullMax", _p(max(nulls)))
            macros.add(f"{prefix}FitWithinFifty", str(sum(b <= 50 for b in firsts)))
            if (p.tag, model) == ("Harm", "M2b_pca_reconstruction"):
                # The text: the two fits with the lowest AUC-PR, the reported one first, give the
                # two earliest first detections.
                by_auc = sorted(runs, key=lambda r: r["auc_pr"])
                if by_auc[0]["seed"] != 42 or {
                    r["budget_for_first_detection"] for r in by_auc[:2]
                } != set(sorted(firsts)[:2]):
                    raise ValueError(
                        "PCA's AUC-PR and first detection no longer run opposite ways."
                    )
            rows.append(
                " & ".join(
                    [
                        {"Harm": "harmonised", "Temp": "temporal"}[p.tag],
                        LABELS[model],
                        f"[${_sci(min(aucs))}$, ${_sci(max(aucs))}$]",
                        f"{_int(float(np.median(firsts)))} [{_int(min(firsts))}, {_int(max(firsts))}]",
                        f"{_p(float(np.median(nulls)))} [{_p(min(nulls))}, {_p(max(nulls))}]",
                        f"[{_pct(min(at_500))}, {_pct(max(at_500))}]",
                    ]
                )
                + r" \\"
            )
        macros.add(f"{p.tag}FitSeeds", str(len(next(iter(variance.values()))["runs"])))
    _write("table_fit_variance.tex", rows)


def _campaign_accounts(p: Protocol) -> dict[int, str]:
    """campaign_id -> source user, the account the run's bootstrap resamples (`ledger.campaign_users`)."""
    name = {"Harm": "harmonised", "Temp": "temporal"}[p.tag]
    with (USER_DAYS / f"positives_{name}.csv").open(encoding="utf-8") as handle:
        accounts = {
            p.event_campaign[int(r["event_id"])]: r["src_user"] for r in csv.DictReader(handle)
        }
    if len(accounts) != p.campaigns or len(set(accounts.values())) != p.accounts:
        raise ValueError(f"{p.tag}: campaigns or accounts differ from the run's budget summary")
    return accounts


def _recall_interval(
    p: Protocol, model: str, accounts: dict[int, str], k: int, n_resamples: int = 2_000
) -> tuple[float, float]:
    """95% percentile interval of campaign recall at k over accounts resampled with replacement."""
    best: dict[int, int] = {}
    for campaign, _, rank in p.ledger[model]:
        best[campaign] = min(rank, best.get(campaign, rank))
    if abs(np.mean([b <= k for b in best.values()]) - p.recall(model, k)) > 1e-9:
        raise ValueError(f"{p.tag} {model}: ledger recall at {k} differs from the run's")
    users = sorted(set(accounts.values()))
    detected = np.zeros(len(users))
    size = np.zeros(len(users))
    for campaign, b in best.items():
        u = users.index(accounts[campaign])
        detected[u] += b <= k
        size[u] += 1
    rng = np.random.default_rng(k)
    draws = rng.integers(0, len(users), (n_resamples, len(users)))
    share = detected[draws].sum(axis=1) / size[draws].sum(axis=1)
    return float(np.quantile(share, 0.025)), float(np.quantile(share, 0.975))


def _budgets_table(macros: Macros, p: Protocol) -> None:
    """The three smallest campaign budgets, budget quantiles and, for a partial first test day,
    the first detection without it (harmonised: test day 1 holds only hours 41 to 47)."""
    t = p.tag
    partial = (p.day_rows[0]["day"],) if t == "Harm" else ()
    later = [(r["events"], r["positives"]) for r in p.day_rows if r["day"] not in partial]
    if partial:
        sizes = [r["events"] for r in p.day_rows[1:]]
        macros.add(f"{t}DayOneEvents", _int(p.day_rows[0]["events"]))
        macros.add(f"{t}OtherDaysMin", _int(min(sizes)))
        macros.add(f"{t}OtherDaysMax", _int(max(sizes)))
    accounts = _campaign_accounts(p)
    rows = []
    for model in sorted(p.non_trivial(), key=p.first_detection):
        budgets = p.campaign_budgets(model)
        stats = p.budgets[model]["statistics"]
        median = stats["budget_q50"]
        low, high = _recall_interval(p, model, accounts, 500)
        cells = [
            LABELS[model],
            # b* is in the results tables; the next two say how many campaigns it rests on.
            ", ".join(_int(b) for b in budgets[1:3]),
            _int(stats["budget_q10"]["value"]),
            f"{_int(median['value'])} [{_int(median['ci_low'])}, {_int(median['ci_high'])}]",
            f"{_pct(p.recall(model, 500))} [{_pct(low)}, {_pct(high)}]"
            if p.recall(model, 500)
            else _pct(0.0),
        ]
        prefix = t + MACRO_NAMES[model]
        macros.add(f"{prefix}SecondBudget", _int(budgets[1]))
        macros.add(f"{prefix}ThirdBudget", _int(budgets[2]))
        macros.add(f"{prefix}MedianBudget", _int(stats["budget_q50"]["value"]))
        if partial:
            without = p.campaign_budgets(model, partial)[0]
            null = 1 - random_first_detection_survival(later, without)
            macros.add(f"{prefix}FirstWithoutDayOne", _int(without))
            macros.add(f"{prefix}NullWithoutDayOne", _p(null))
            cells += [_int(without), _p(null)]
        rows.append(" & ".join(cells) + r" \\")
    _write(f"table_budgets_{t.lower()}.tex", rows)


def _intrusion_table(macros: Macros, p: Protocol) -> None:
    """The first test day on which some red-team event enters the top k of its day.

    With one intrusion, this is the operational question: on which day is the operation first
    noticed at a given staffing. The null is exact: a random ranking puts no positive of day d
    in its top k with probability C(N_d - k, p_d) / C(N_d, p_d), independently across days.
    """
    t = p.tag
    budgets = (50, 100, 500)
    attack_day = min(r["day"] for r in p.day_rows if r["positives"])
    macros.add(f"{t}AttackFirstDay", str(attack_day))
    rows = []
    for model in sorted(p.non_trivial() + ["M0a_random"], key=p.first_detection):
        cells = [LABELS[model]]
        for k in budgets:
            days = sorted({day for _, day, rank in p.ledger[model] if rank <= k})
            if not days:
                cells.append("none")
                continue
            first = days[0]
            upto = [(r["events"], r["positives"]) for r in p.day_rows if r["day"] <= first]
            null = 1 - random_first_detection_survival(upto, k)
            cells.append(f"{first} ({_p(null)})")
            macros.add(f"{t}{MACRO_NAMES[model]}IntrusionDay{BUDGET_NAMES[k]}", str(first))
            macros.add(f"{t}{MACRO_NAMES[model]}IntrusionNull{BUDGET_NAMES[k]}", _p(null))
        rows.append(" & ".join(cells) + r" \\")
    _write(f"table_intrusion_{t.lower()}.tex", rows)


def _misc(macros: Macros, harm: Protocol, temp: Protocol) -> None:
    tuning = _load(temp.tables / "tuning.json")
    for entry in tuning:
        macros.add(f"TuneCount{MACRO_NAMES[entry['model']]}", str(len(entry["candidates"])))
    for p in (harm, temp):
        macros.add(
            f"{p.tag}NullWithinTen", _pct(1 - random_first_detection_survival(p.days, 10), 2)
        )
        macros.add(f"{p.tag}ZeroRecallUpper", _pct(1 - 0.05 ** (1 / p.campaigns)))
        best, worst = p.budgets["M2a_pair_rarity"]["tie_range_first_detection"]
        macros.add(f"{p.tag}PairRarityTieBest", _int(best))
        macros.add(f"{p.tag}PairRarityTieWorst", _int(worst))
        macros.add(
            f"{p.tag}PairRarityTieBestNull",
            _p(1 - random_first_detection_survival(p.days, best)),
        )
        macros.add(
            f"{p.tag}PairRarityTieWorstNull",
            _p(1 - random_first_detection_survival(p.days, worst)),
        )
        significant_floor = sum(c["significant"] and c["at_resolution_floor"] for c in p.pairs)
        macros.add(f"{p.tag}PairsAtFloor", str(significant_floor))
    for p in (harm, temp):
        sizes: dict[int, int] = defaultdict(int)
        per_day: dict[int, set[int]] = defaultdict(set)
        for campaign, day, _ in p.ledger["M0a_random"]:
            sizes[campaign] += 1
            per_day[day].add(campaign)
        busiest = sorted(per_day.items(), key=lambda kv: -len(kv[1]))[:3]
        macros.add(f"{p.tag}CampaignSingletons", str(sum(n == 1 for n in sizes.values())))
        macros.add(f"{p.tag}CampaignLargest", str(max(sizes.values())))
        macros.add(
            f"{p.tag}BusiestDays",
            ", ".join(f"{day} ({len(c)})" for day, c in busiest),
        )
        macros.add(
            f"{p.tag}BusiestDaysCampaigns",
            str(len(set().union(*(c for _, c in busiest)))),
        )
    # Larroche (2026), Table 4: the harmonised LANL universe, 369,569,626 events, 702 malicious.
    log = (RUN / "evaluate_harmonised.log").read_text(encoding="utf-8")
    train = int(re.search(r"sample of \d+/(\d+) training rows", log).group(1))  # type: ignore[union-attr]
    total = train + sum(n for n, _ in harm.days)
    if (total, sum(k for _, k in harm.days)) != (LARROCHE_EVENTS, LARROCHE_MALICIOUS):
        raise ValueError("The harmonised universe no longer equals Larroche's (2026) Table 4.")
    macros.add("HarmUniverseEvents", _int(total))


def _larroche_table(harm: Protocol) -> None:
    """Event-level AUC and AP in percent on Larroche's (2026) universe: his Table 5, then ours."""
    rows = [
        f"{name} \\parencite{{larroche2026fair}} & {auc} & {auc_rep} & {ap} & {ap_rep} \\\\"
        for name, (auc, auc_rep, ap, ap_rep) in LARROCHE_TABLE5.items()
    ]
    rows.append(r"\midrule")
    for model in sorted(harm.non_trivial(), key=lambda m: -harm.metric(m, "auc_pr")):
        rows.append(
            f"{LABELS[model]} (this study) & {100 * harm.metric(model, 'roc_auc'):.2f} & & "
            f"{100 * harm.metric(model, 'auc_pr'):.3f} & \\\\"
        )
    _write("table_larroche.tex", rows)


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
    # Columns of a run line: model, TPR, FPR, TP, FP, F1, AUC, AP, ...
    run_auc = [float(r[6]) for r in runs]
    run_ap = [float(r[7]) for r in runs]
    # The published value is one number, not a population mean: each sample is set against it
    # through the 95% prediction interval for one further run, AP also on the log scale.
    quantile = float(student_t.ppf(0.975, n - 1))
    for name, sample, published in (
        ("Auc", run_auc, EULER_PUBLISHED["auc"]),
        ("Ap", run_ap, EULER_PUBLISHED["ap"]),
    ):
        values_ = np.array(sample)
        if not (values_ < published).all():
            raise ValueError(f"The text says every authors' run is below the published {name}.")
        macros.add(f"EulerAuthors{name}Min", f"{values_.min():.4f}")
        macros.add(f"EulerAuthors{name}Max", f"{values_.max():.4f}")
        macros.add(f"EulerAuthors{name}Gap", f"{published - values_.mean():.4f}")
        for scale, x in (("", values_), ("Log", np.log(values_))):
            half = quantile * x.std(ddof=1) * math.sqrt(1 + 1 / n)
            low, high = x.mean() - half, x.mean() + half
            if scale:
                low, high = math.exp(low), math.exp(high)
            macros.add(f"EulerAuthors{name}{scale}PredLow", f"{low:.4f}")
            macros.add(f"EulerAuthors{name}{scale}PredHigh", f"{high:.4f}")
    tpr = float(values["mean"]["TPR"])
    fpr = float(values["mean"]["FPR"])
    precision = float(np.mean([float(r[3]) / (float(r[3]) + float(r[4])) for r in runs]))
    macros.add("EulerAuthorsTpr", _pct(tpr))
    macros.add("EulerAuthorsFpr", _pct(fpr, 3))
    macros.add("EulerAuthorsPrecision", f"{precision:.4f}")
    macros.add("EulerPublishedTpr", _pct(EULER_PUBLISHED["tpr"]))
    macros.add("EulerPublishedFpr", _pct(EULER_PUBLISHED["fpr"], 3))
    macros.add("EulerPublishedPrecision", f"{EULER_PUBLISHED['precision']:.4f}")
    macros.add("EulerPublishedNodes", _int(EULER_PUBLISHED["nodes"]))
    macros.add("EulerPublishedEvents", _int(EULER_PUBLISHED["events"]))
    notes = (RUN / "REVIEW_NOTES.md").read_text(encoding="utf-8")
    nodes = re.search(r"([\d,]+) nodes in its node map", notes)
    kept = re.search(r'keeps lines containing "NTLM": ([\d,]+) events, ([\d,]+) without', notes)
    assert nodes and kept
    macros.add("EulerAuthorsNodes", nodes.group(1).replace(",", "{,}"))
    macros.add("EulerAuthorsNtlmLines", kept.group(1).replace(",", "{,}"))
    macros.add("EulerAuthorsNtlmNoLoops", kept.group(2).replace(",", "{,}"))
    macros.add("EulerAuthorsRuns", str(n))
    macros.add("EulerAuthorsAuc", f"{auc:.4f}")
    macros.add("EulerAuthorsAucSd", f"{auc_sd:.4f}")
    macros.add("EulerAuthorsAp", f"{ap:.4f}")
    macros.add("EulerAuthorsApSd", f"{ap_sd:.4f}")
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
            ax.axhspan(10, 500, color="0.93", zorder=0)
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
                0.98, 0.03, "10 to 500 alerts/day", transform=ax.transAxes, fontsize=6, ha="right"
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


def _random_recall_band(
    p: Protocol, grid: np.ndarray, n_simulations: int = 1_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Campaign recall under uniformly random rankings of each day: mean and central 95% range.

    The curve the paper compares with, rather than the one realisation of the random floor M0a.
    """
    positives = p.ledger["M0a_random"]
    campaigns = sorted({c for c, _, _ in positives})
    position = {c: i for i, c in enumerate(campaigns)}
    index = np.array([position[c] for c, _, _ in positives])
    by_day: dict[int, list[int]] = defaultdict(list)
    for i, (_, day, _) in enumerate(positives):
        by_day[day].append(i)
    rng = np.random.default_rng(0)
    curves = np.empty((n_simulations, grid.size))
    ranks = np.empty(len(positives), dtype=np.int64)
    for s in range(n_simulations):
        for day, rows in by_day.items():
            n = p.days_by_number[day]["events"]
            ranks[rows] = rng.choice(n, size=len(rows), replace=False) + 1
        budgets = np.full(len(campaigns), np.iinfo(np.int64).max)
        np.minimum.at(budgets, index, ranks)
        curves[s] = (budgets[None, :] <= grid[:, None]).mean(axis=1)
    return (
        curves.mean(axis=0),
        np.quantile(curves, 0.025, axis=0),
        np.quantile(curves, 0.975, axis=0),
    )


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
        mean, low, high = _random_recall_band(p, grid)
        ax.fill_between(grid, low, high, step="post", color="0.82", linewidth=0, zorder=0.5)
        ax.step(
            grid,
            mean,
            where="post",
            color="0.45",
            linewidth=1.2,
            linestyle=(0, (1, 1)),
            label="random ranking (mean, 95% band)",
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
    _results_table(macros, "table_results_harmonised.tex", harm)
    _results_table(macros, "table_results_temporal.tex", temp)
    _regime_table(macros, temp, temp_two)
    for p in (harm, temp):
        _units_table(macros, p)
        _fixed_table(p)
        _budgets_table(macros, p)
        _intrusion_table(macros, p)
    _misc(macros, harm, temp)
    _larroche_table(harm)
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
