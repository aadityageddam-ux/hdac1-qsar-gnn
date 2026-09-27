"""Phase 6b: the two diagnostics that explain a result rather than just reporting one.

1. **Learning curve.** Test error against training-set size, for both models. This is
   what separates "the GNN is architecturally worse here" from "the GNN is data-limited
   here", and those are very different conclusions. Subsampling is done BY SCAFFOLD
   GROUP, never by compound: drawing individual compounds would put members of the same
   scaffold group back on both sides and quietly reintroduce the leakage the split
   exists to prevent.

2. **Random-split control.** Both models refit under a seeded random split at the same
   fold fractions. The gap between the two numbers is the optimism that a random-split
   evaluation would have bought, measured rather than asserted.

This script retrains models and takes a while, which is exactly why it is separate from
06_compare.py - the headline table stays reproducible in seconds.
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
RANDOM_SPLIT_SEED = 20260925

LABEL_RF = "Random forest (ECFP4 + descriptors)"
LABEL_GNN = "GINE graph neural network"


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

    print("[1/3] Featurising and graphing the fixed folds ...")
    X_val, _ = F.build_features(val.canonical_smiles.tolist())
    X_test, _ = F.build_features(test.canonical_smiles.tolist())
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    val_graphs, _ = G.smiles_to_graphs(val.canonical_smiles.tolist(), val.pIC50.to_numpy())
    test_graphs, _ = G.smiles_to_graphs(test.canonical_smiles.tolist(), y_test)
    print(f"      val {len(val)}, test {len(test)}")

    print("[2/3] Learning curve (subsampling whole scaffold groups) ...")
    curve: dict[str, dict[str, list[float]]] = {
        LABEL_RF: {"mean": [], "sd": [], "n_train": []},
        LABEL_GNN: {"mean": [], "sd": [], "n_train": []},
    }
    curve_detail = []
    for fraction in FRACTIONS:
        rf_scores, gnn_scores, sizes = [], [], []
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

        for label, scores in ((LABEL_RF, rf_scores), (LABEL_GNN, gnn_scores)):
            curve[label]["mean"].append(float(np.mean(scores)))
            curve[label]["sd"].append(float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
            curve[label]["n_train"].append(float(np.mean(sizes)))
        curve_detail.append({
            "fraction": fraction, "n_train_mean": float(np.mean(sizes)),
            "rf_rmse": rf_scores, "gnn_rmse": gnn_scores,
        })
        print(f"      {fraction:5.0%}  n~{np.mean(sizes):6.0f}   "
              f"RF {np.mean(rf_scores):.4f}+-{np.std(rf_scores, ddof=1):.4f}   "
              f"GNN {np.mean(gnn_scores):.4f}+-{np.std(gnn_scores, ddof=1):.4f}")

    print("[3/3] Random-split control ...")
    rng = np.random.default_rng(RANDOM_SPLIT_SEED)
    perm = rng.permutation(len(split))
    n_tr, n_va = len(train), len(val)
    r_train = split.iloc[perm[:n_tr]]
    r_val = split.iloc[perm[n_tr:n_tr + n_va]]
    r_test = split.iloc[perm[n_tr + n_va:]]
    ry_test = r_test.pIC50.to_numpy(dtype=np.float64)

    rX_fit, _ = F.build_features(
        r_train.canonical_smiles.tolist() + r_val.canonical_smiles.tolist()
    )
    rX_test, _ = F.build_features(r_test.canonical_smiles.tolist())
    ry_fit = np.concatenate([r_train.pIC50.to_numpy(), r_val.pIC50.to_numpy()])
    rf_random = rf.fit_predict_multiseed(
        rX_fit.astype(np.float32), ry_fit, rX_test.astype(np.float32),
        best_rf_config, y_eval=ry_test, seeds=(0,), verbose=False,
    )
    rf_random_rmse = ev.compute_metric("rmse", ry_test, rf_random.mean_prediction)

    r_train_graphs, _ = G.smiles_to_graphs(r_train.canonical_smiles.tolist(), r_train.pIC50.to_numpy())
    r_val_graphs, _ = G.smiles_to_graphs(r_val.canonical_smiles.tolist(), r_val.pIC50.to_numpy())
    r_test_graphs, _ = G.smiles_to_graphs(r_test.canonical_smiles.tolist(), ry_test)
    gmodel, gtr = G.train_model(
        r_train_graphs + r_val_graphs, None, G.GNNConfig(**best_gnn_config),
        seed=0, fixed_epochs=best_epoch,
    )
    gnn_random_rmse = ev.compute_metric("rmse", ry_test, G.predict(gmodel, r_test_graphs, gtr))

    scaffold_rf = rf_metrics["regression"]["rmse"]
    scaffold_gnn = gnn_metrics["regression"]["rmse"]
    print(f"      RF   random {rf_random_rmse:.4f}  vs scaffold {scaffold_rf:.4f}  "
          f"(optimism {scaffold_rf - rf_random_rmse:+.4f})")
    print(f"      GNN  random {gnn_random_rmse:.4f}  vs scaffold {scaffold_gnn:.4f}  "
          f"(optimism {scaffold_gnn - gnn_random_rmse:+.4f})")

    P.save_figure(
        P.learning_curve(
            FRACTIONS,
            {LABEL_RF: (curve[LABEL_RF]["mean"], curve[LABEL_RF]["sd"]),
             LABEL_GNN: (curve[LABEL_GNN]["mean"], curve[LABEL_GNN]["sd"])},
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
        "random_split_control": {
            "seed": RANDOM_SPLIT_SEED,
            "fold_sizes": {"train": int(n_tr), "val": int(n_va), "test": int(len(r_test))},
            LABEL_RF: {"random_rmse": rf_random_rmse, "scaffold_rmse": scaffold_rf,
                       "optimism": scaffold_rf - rf_random_rmse},
            LABEL_GNN: {"random_rmse": gnn_random_rmse, "scaffold_rmse": scaffold_gnn,
                        "optimism": scaffold_gnn - gnn_random_rmse},
            "note": (
                "Optimism is scaffold RMSE minus random RMSE: how much better a "
                "random-split evaluation would have looked on identical data and models."
            ),
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"      wrote diagnostics.json and fig_learning_curve.png")
    print(f"07_diagnostics.py complete in {payload['runtime_seconds']:.0f}s.")


if __name__ == "__main__":
    main()
