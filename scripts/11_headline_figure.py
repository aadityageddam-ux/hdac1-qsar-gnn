"""Headline figure: how much a random split flatters each model, on three targets.

Reads results/multitarget.json (written by 10_multitarget.py) and plots two things:
test RMSE under each split for the random forest and Chemprop, and the "optimism"
(scaffold RMSE minus random RMSE) of each model per target.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RF, CP = "#4c78a8", "#e4572e"

data = json.loads((ROOT / "results" / "multitarget.json").read_text())["targets"]
targets = list(data)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.7, 1]})

# A: test RMSE by split, per target (lower is better)
labels, rf_vals, cp_vals = [], [], []
for t in targets:
    for split in ("scaffold_split", "random_split"):
        labels.append(f"{t}\n{'scaffold' if split.startswith('s') else 'random'}")
        rf_vals.append(data[t][split]["random_forest"]["rmse"])
        cp_vals.append(data[t][split]["chemprop"]["rmse"])
x = list(range(len(labels)))
# Dots, not bars: the axis does not start at zero, and bars would exaggerate small gaps.
for i in x:
    ax_a.plot([i, i], [rf_vals[i], cp_vals[i]], color="#bbbbbb", lw=2, zorder=1)
ax_a.scatter(x, rf_vals, s=70, color=RF, label="Random forest (ECFP4)", zorder=2)
ax_a.scatter(x, cp_vals, s=70, color=CP, marker="s", label="Chemprop D-MPNN", zorder=2)
ax_a.set_xticks(x, labels, fontsize=9)
ax_a.set_ylabel("test RMSE (pIC50), lower is better")
ax_a.set_ylim(0.5, 0.76)
ax_a.set_title("Which model wins depends on the split", fontsize=11)
ax_a.legend(frameon=False, fontsize=9, loc="upper right")

# B: optimism (how much a random split flatters each model)
w = 0.38
bx = range(len(targets))
rf_opt = [data[t]["optimism"]["random_forest"] for t in targets]
cp_opt = [data[t]["optimism"]["chemprop"] for t in targets]
ax_b.bar([i - w / 2 for i in bx], rf_opt, w, color=RF)
ax_b.bar([i + w / 2 for i in bx], cp_opt, w, color=CP)
for i, t in enumerate(targets):
    ax_b.text(i + w / 2, cp_opt[i] + 0.004, f"{data[t]['optimism_ratio_rf_over_chemprop']:.2f}x", ha="center", fontsize=8, color="#444")
ax_b.set_xticks(list(bx), targets)
ax_b.set_ylabel("optimism = scaffold RMSE − random RMSE")
ax_b.set_ylim(0, 0.21)
ax_b.set_title("A random split flatters the graph model more\n(label: forest optimism / Chemprop optimism)", fontsize=11)

for ax in (ax_a, ax_b):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

fig.tight_layout()
out = ROOT / "results" / "fig_headline.png"
fig.savefig(out, dpi=160)
print("wrote", out)
