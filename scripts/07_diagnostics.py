"""The learning curve: test error against training-set size.

This is what separates "the graph models are worse here" from "the graph models are
data-limited here", which are very different conclusions.

Subsampling is done by whole scaffold group, never by compound. Drawing individual
compounds would put members of the same group back on both sides and quietly reintroduce
exactly the leakage the split exists to stop.

The random-split control used to live here and now sits in 07b_random_split_control.py.
The curve retrains at five training-set sizes and runs for hours; the control is a handful
of fits and has already needed recomputing twice, so it's better off separate.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline_rf as rf  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import featurize as F  # noqa: E402
from src import chemprop_model as CP  # noqa: E402
from src import gnn as G  # noqa: E402
from src import plotting as P  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

FRACTIONS: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00)

# The learning curve runs a REDUCED protocol compared with the headline result, and says so
# rather than implying otherwise. Measured cost on this machine is ~5 s/epoch for the
# selected architecture on the full training fold, so the curve at the headline protocol
# (5 fractions x 2 models x 5 seeds, early stopping with patience 40 and a 300-epoch cap)
# would run for roughly four hours to produce a trend line.
#
# Two seeds give a visible band, and the epoch cap below bounds the worst case. This is a
# defensible economy for a diagnostic whose question is "which way does the curve point",
# not "what is the exact RMSE at 50% of the data" - but it is a real difference from the
# headline protocol and is recorded in diagnostics.json alongside the numbers.
CURVE_SEEDS: tuple[int, ...] = (0, 1)
CURVE_MAX_EPOCHS = 120
CURVE_PATIENCE = 20

LABEL_RF = "Random forest (ECFP4 + descriptors)"
LABEL_GNN = "GINE graph neural network"
LABEL_CHEMPROP = "Chemprop D-MPNN"


def subsample_by_scaffold_group(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Take whole scaffold groups until ``fraction`` of the compounds is reached.

    Sampling whole groups is the point. Sampling individual compounds would split a
    scaffold group across the retained and discarded sets, which changes what the model
    is being asked to generalise over and makes the curve incomparable to the headline
    result.
    """
    if fraction >= 1.0:
        return frame
    rng = np.random.default_rng(seed)
    groups = frame.scaffold_group_id.unique()
    rng.shuffle(groups)
    target = int(round(len(frame) * fraction))
    taken, total = [], 0
    for g in groups:
        members = frame[frame.scaffold_group_id == g]
        taken.append(g)
        total += len(members)
        if total >= target:
            break
    return frame[frame.scaffold_group_id.isin(taken)]


def main() -> None:
    t_start = time.time()
    rf_metrics = json.loads((RESULTS / "metrics_rf.json").read_text(encoding="utf-8"))
    gnn_metrics = json.loads((RESULTS / "metrics_gnn.json").read_text(encoding="utf-8"))
    best_rf_config = rf_metrics["tuning"]["best_config"]
    best_gnn_config = gnn_metrics["config"]
    best_epoch = int(gnn_metrics["tuning"]["best"]["best_epoch"])

    split = pd.read_csv(DATA / "split_assignment.csv")
    train = split[split.fold == "train"].reset_index(drop=True)
    val = split[split.fold == "val"].reset_index(drop=True)
    test = split[split.fold == "test"].reset_index(drop=True)
    y_test = test.pIC50.to_numpy(dtype=np.float64)

    print("[1/1] Featurising and graphing the fixed folds ...")
    X_val, _ = F.build_features(val.canonical_smiles.tolist())
    X_test, _ = F.build_features(test.canonical_smiles.tolist())
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    val_graphs, _ = G.smiles_to_graphs(val.canonical_smiles.tolist(), val.pIC50.to_numpy())
    test_graphs, _ = G.smiles_to_graphs(test.canonical_smiles.tolist(), y_test)
    print(f"      val {len(val)}, test {len(test)}")

    print("[1/2] Learning curve (subsampling whole scaffold groups) ...")
    curve: dict[str, dict[str, list[float]]] = {
        label: {"mean": [], "sd": [], "n_train": []}
        for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP)
    }
    curve_detail = []
    for fraction in FRACTIONS:
        rf_scores, gnn_scores, cp_scores, sizes = [], [], [], []
        for seed in CURVE_SEEDS:
            subset = subsample_by_scaffold_group(train, fraction, seed)
            sizes.append(len(subset))
            smiles = subset.canonical_smiles.tolist()
            y_sub = subset.pIC50.to_numpy(dtype=np.float64)

            X_sub, _ = F.build_features(smiles)
            res = rf.fit_predict_multiseed(
                X_sub.astype(np.float32), y_sub, X_test, best_rf_config,
                y_eval=y_test, seeds=(seed,), verbose=False,
            )
            rf_scores.append(ev.compute_metric("rmse", y_test, res.mean_prediction))

            sub_graphs, _ = G.smiles_to_graphs(smiles, y_sub)
            curve_config = G.GNNConfig(
                **{**best_gnn_config, "max_epochs": CURVE_MAX_EPOCHS, "patience": CURVE_PATIENCE}
            )
            model, tr = G.train_model(sub_graphs, val_graphs, curve_config, seed=seed)
            gnn_scores.append(ev.compute_metric("rmse", y_test, G.predict(model, test_graphs, tr)))

            cp_model, _, _, _ = CP.train_once(
                subset, val, seed=seed, max_epochs=CURVE_MAX_EPOCHS, use_early_stopping=True
            )
            cp_scores.append(ev.compute_metric("rmse", y_test, CP.predict(cp_model, test)))

        for label, scores in (
            (LABEL_RF, rf_scores), (LABEL_GNN, gnn_scores), (LABEL_CHEMPROP, cp_scores)
        ):
            curve[label]["mean"].append(float(np.mean(scores)))
            curve[label]["sd"].append(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
            curve[label]["n_train"].append(float(np.mean(sizes)))
        curve_detail.append({
            "fraction": fraction, "n_train_mean": float(np.mean(sizes)),
            "rf_rmse": rf_scores, "gnn_rmse": gnn_scores, "chemprop_rmse": cp_scores,
        })
        print(f"      {fraction:5.0%}  n~{np.mean(sizes):6.0f}   "
              f"RF {np.mean(rf_scores):.4f}+-{np.std(rf_scores, ddof=1):.4f}   "
              f"GINE {np.mean(gnn_scores):.4f}+-{np.std(gnn_scores, ddof=1):.4f}   "
              f"Chemprop {np.mean(cp_scores):.4f}+-{np.std(cp_scores, ddof=1):.4f}")

    P.save_figure(
        P.learning_curve(
            FRACTIONS,
            {label: (curve[label]["mean"], curve[label]["sd"])
             for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP)},
            n_train_total=len(train),
            title=f"Learning curve (scaffold-group subsampling, {len(CURVE_SEEDS)} seeds, capped epochs)",
        ),
        RESULTS / "fig_learning_curve.png",
    )

    payload = {
        "learning_curve": {
            "fractions": list(FRACTIONS), "seeds": list(CURVE_SEEDS),
            "n_train_total": int(len(train)), "detail": curve_detail,
            LABEL_RF: curve[LABEL_RF], LABEL_GNN: curve[LABEL_GNN],
            LABEL_CHEMPROP: curve[LABEL_CHEMPROP],
            "subsampling": "whole scaffold groups, never individual compounds",
            "reduced_protocol": {
                "n_seeds": len(CURVE_SEEDS),
                "gnn_max_epochs": CURVE_MAX_EPOCHS,
                "gnn_patience": CURVE_PATIENCE,
            "note": (
                    "The curve uses 2 seeds and a capped epoch budget, unlike the headline "
                    "result which uses 5 seeds and patience 40 with a 300-epoch cap. The "
                    "curve answers which way performance is trending with data volume; it "
                    "is not a second estimate of the headline RMSE."
                ),
            },
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    # Merge rather than overwrite: 07b_random_split_control.py owns the control section.
    out = RESULTS / "diagnostics.json"
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    existing.update(payload)
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"      wrote learning_curve into diagnostics.json and fig_learning_curve.png")
    print(f"07_diagnostics.py complete in {payload['runtime_seconds']:.0f}s.")


if __name__ == "__main__":
    main()
