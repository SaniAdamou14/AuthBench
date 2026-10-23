"""Per-rule view of M1u on the temporal test split: why its ROC-AUC sits below 0.5.

Fits M1u exactly as the pipeline does (UniformRulesScorer() on the train split), then scores
every malicious test event and a seeded 1-in-25 hash sample of the benign ones. A uniform
benign sample leaves ROC-AUC unbiased (it does not depend on the class ratio).
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from authbench.models.rules import RULES, UniformRulesScorer

store = Path(sys.argv[1])
train = pl.scan_parquet(store / "train" / "*.parquet")
test = pl.scan_parquet(store / "test" / "*.parquet")

model = UniformRulesScorer()
model.fit(train)
sample = test.filter(pl.col("is_malicious") | (pl.col("event_id").hash(seed=0) % 25 == 0))
frame = sample.collect()
labels = frame["is_malicious"].to_numpy()
print(f"sample: {int(labels.sum())} malicious, {int((~labels).sum())} benign (1 in 25)")

scores = model.rule_scores(frame.lazy())
assert scores["event_id"].to_list() == frame["event_id"].to_list()
combined = model.score(frame.lazy()).to_numpy()
print(f"M1u ROC-AUC on the sample: {roc_auc_score(labels, combined):.4f}")

print(f"{'rule':<28}{'mean benign':>12}{'mean malicious':>16}{'ROC-AUC':>9}")
for rule in RULES:
    values = scores[rule.id].to_numpy()
    print(
        f"{rule.id:<28}{values[~labels].mean():>12.4f}{values[labels].mean():>16.4f}"
        f"{roc_auc_score(labels, values):>9.4f}"
    )

machine = frame["src_user_is_machine"].to_numpy()
print(f"machine accounts: {machine[~labels].mean():.3f} of benign, {machine[labels].mean():.3f} of malicious")
