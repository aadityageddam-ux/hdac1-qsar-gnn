"""How much apparent accuracy the evaluation protocol buys you, for free.

Refits every model under a seeded random split at the same fold sizes as the scaffold
split and reports the difference. That difference - optimism - is what a random-split
evaluation adds on identical data with identical models.

Kept separate from the learning curve because it's minutes rather than hours, and because
it has already had to be recomputed twice.

Two things this gets right that earlier versions didn't, both of which reversed the
headline finding when they were wrong:

1. Each split picks its own epoch count on its own validation fold. Reusing the
   scaffold-selected count under-trains the neural models on the easier split.
2. Both sides of the subtraction use the same seed protocol. Seed-ensembling is worth
   ~0.04 RMSE to a neural model and ~0.001 to a forest, so ensembling only one side
   strips optimism from the neural models and nothing from the forest.

The forest has no epoch schedule and gains nothing from seed averaging, so any protocol
detail applied unevenly lands entirely on the neural models. Both bugs did. See
AI_USAGE.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline_rf as rf  # noqa: E402
from src import chemprop_model as CP  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import featurize as F  # noqa: E402
from src import gnn as G  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

RANDOM_SPLIT_SEED = 20260925
LABEL_RF = "Random forest (ECFP4 + descriptors)"
LABEL_GNN = "GINE graph neural network"
LABEL_CHEMPROP = "Chemprop D-MPNN"


def recompute_from_cache() -> None:
    """Re-derive the control from values already measured, without retraining anything.

    The random-split RMSEs are real measurements already recorded in diagnostics.json and
    are not changed here; only the scaffold side of the subtraction is corrected to match
    their seed protocol. Retraining would cost eighteen hours and would not change a
    measurement.
    """
    out = RESULTS / "diagnostics.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    control = payload["random_split_control"]

    rf_metrics = json.loads((RESULTS / "metrics_rf.json").read_text(encoding="utf-8"))
    gnn_metrics = json.loads((RESULTS / "metrics_gnn.json").read_text(encoding="utf-8"))
    cp_metrics = json.loads((RESULTS / "metrics_chemprop.json").read_text(encoding="utf-8"))
    single = {
        LABEL_RF: rf_metrics["multiseed"]["per_seed_rmse_mean"],
        LABEL_GNN: gnn_metrics["per_seed_rmse_mean"],
        LABEL_CHEMPROP: cp_metrics["per_seed_rmse_mean"],
    }
    ensembled = {
        LABEL_RF: rf_metrics["regression"]["rmse"],
        LABEL_GNN: gnn_metrics["regression"]["rmse"],
        LABEL_CHEMPROP: cp_metrics["regression"]["rmse"],
    }
    for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP):
        block = control[label]
        block["scaffold_rmse"] = single[label]
        block["scaffold_rmse_ensembled"] = ensembled[label]
        block["optimism"] = single[label] - block["random_rmse"]
        block["n_seeds_each_side"] = 1
    control["comparison"] = (
        "single model on both sides: the per-seed mean scaffold RMSE against a single-seed "
        "random-split RMSE. The ensembled scaffold RMSE is recorded for reference but is NOT "
        "used for optimism, because the random side is not ensembled and seed-ensembling "
        "helps the neural models far more than the forest."
    )
    control["recomputed_from_cache"] = True
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{'model':38s} {'random':>8s} {'scaffold':>9s} {'optimism':>9s}")
    for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP):
        b = control[label]
        print(f"{label:38s} {b['random_rmse']:8.4f} {b['scaffold_rmse']:9.4f} {b['optimism']:+9.4f}")
    print()
    print("re-derived the control from cached measurements; nothing was retrained")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute", action="store_true",
        help="re-derive optimism from already-measured values instead of retraining.",
    )
    if parser.parse_args().recompute:
        recompute_from_cache()
        return

    t_start = time.time()

    rf_metrics = json.loads((RESULTS / "metrics_rf.json").read_text(encoding="utf-8"))
    gnn_metrics = json.loads((RESULTS / "metrics_gnn.json").read_text(encoding="utf-8"))
    cp_metrics = json.loads((RESULTS / "metrics_chemprop.json").read_text(encoding="utf-8"))
    best_rf_config = rf_metrics["tuning"]["best_config"]
    best_gnn_config = gnn_metrics["config"]

    split = pd.read_csv(DATA / "split_assignment.csv")
    n_tr = int((split.fold == "train").sum())
    n_va = int((split.fold == "val").sum())

    print("[1/4] Building the random split at the same fold sizes ...", flush=True)
    rng = np.random.default_rng(RANDOM_SPLIT_SEED)
    perm = rng.permutation(len(split))
    r_train = split.iloc[perm[:n_tr]]
    r_val = split.iloc[perm[n_tr:n_tr + n_va]]
    r_test = split.iloc[perm[n_tr + n_va:]]
    ry_test = r_test.pIC50.to_numpy(dtype=np.float64)
    ry_fit = np.concatenate([r_train.pIC50.to_numpy(), r_val.pIC50.to_numpy()])
    print(f"      train {len(r_train)}  val {len(r_val)}  test {len(r_test)}", flush=True)

    print("[2/4] Random forest ...", flush=True)
    rX_fit, _ = F.build_features(
        r_train.canonical_smiles.tolist() + r_val.canonical_smiles.tolist()
    )
    rX_test, _ = F.build_features(r_test.canonical_smiles.tolist())
    rf_random = rf.fit_predict_multiseed(
        rX_fit.astype(np.float32), ry_fit, rX_test.astype(np.float32),
        best_rf_config, y_eval=ry_test, seeds=(0,), verbose=False,
    )
    rf_random_rmse = ev.compute_metric("rmse", ry_test, rf_random.mean_prediction)
    print(f"      RF random RMSE {rf_random_rmse:.4f}", flush=True)

    print("[3/4] GINE (epochs selected on the random split's own val fold) ...", flush=True)
    r_train_graphs, _ = G.smiles_to_graphs(
        r_train.canonical_smiles.tolist(), r_train.pIC50.to_numpy()
    )
    r_val_graphs, _ = G.smiles_to_graphs(
        r_val.canonical_smiles.tolist(), r_val.pIC50.to_numpy()
    )
    r_test_graphs, _ = G.smiles_to_graphs(r_test.canonical_smiles.tolist(), ry_test)
    _, g_sel = G.train_model(
        r_train_graphs, r_val_graphs, G.GNNConfig(**best_gnn_config), seed=0
    )
    g_epochs_random = int(g_sel.best_epoch)
    gmodel, gtr = G.train_model(
        r_train_graphs + r_val_graphs, None, G.GNNConfig(**best_gnn_config),
        seed=0, fixed_epochs=g_epochs_random,
    )
    gnn_random_rmse = ev.compute_metric("rmse", ry_test, G.predict(gmodel, r_test_graphs, gtr))
    print(f"      GINE random RMSE {gnn_random_rmse:.4f}  "
          f"(epochs {g_epochs_random} vs {gnn_metrics['tuning']['best']['best_epoch']} on scaffold)",
          flush=True)

    print("[4/4] Chemprop (epochs selected on the random split's own val fold) ...", flush=True)
    cp_epochs_scaffold = int(cp_metrics["config"]["selected_epochs"])
    _, _, _, cp_epochs_random = CP.train_once(
        r_train, r_val, seed=0, max_epochs=CP.MAX_EPOCHS, use_early_stopping=True
    )
    cp_model, _, _, _ = CP.train_once(
        pd.concat([r_train, r_val], ignore_index=True), None, seed=0,
        max_epochs=int(cp_epochs_random), use_early_stopping=False,
    )
    cp_random_rmse = ev.compute_metric("rmse", ry_test, CP.predict(cp_model, r_test))
    print(f"      Chemprop random RMSE {cp_random_rmse:.4f}  "
          f"(epochs {int(cp_epochs_random)} vs {cp_epochs_scaffold} on scaffold)", flush=True)

    # Compare like with like. The random side above is a SINGLE model, so the scaffold side
    # must be a single model too - the per-seed mean, not the five-seed ensembled prediction.
    # Seed-ensembling improves the neural models by ~0.04 RMSE and the forest by ~0.001, so
    # ensembling only one side of the subtraction strips ~0.04 from the neural models'
    # optimism and nothing from the forest's. That is a second protocol asymmetry, distinct
    # from the epoch-selection one, pointing the same way: it makes graph models look less
    # leakage-sensitive than they are.
    scaffold = {
        LABEL_RF: rf_metrics["multiseed"]["per_seed_rmse_mean"],
        LABEL_GNN: gnn_metrics["per_seed_rmse_mean"],
        LABEL_CHEMPROP: cp_metrics["per_seed_rmse_mean"],
    }
    scaffold_ensembled = {
        LABEL_RF: rf_metrics["regression"]["rmse"],
        LABEL_GNN: gnn_metrics["regression"]["rmse"],
        LABEL_CHEMPROP: cp_metrics["regression"]["rmse"],
    }
    random_rmse = {
        LABEL_RF: rf_random_rmse,
        LABEL_GNN: gnn_random_rmse,
        LABEL_CHEMPROP: cp_random_rmse,
    }

    control = {
        "seed": RANDOM_SPLIT_SEED,
        "fold_sizes": {"train": len(r_train), "val": len(r_val), "test": len(r_test)},
        "epochs_selected": {
            "gine_scaffold": int(gnn_metrics["tuning"]["best"]["best_epoch"]),
            "gine_random": g_epochs_random,
            "chemprop_scaffold": cp_epochs_scaffold,
            "chemprop_random": int(cp_epochs_random),
        },
        "comparison": (
            "single model on both sides: the per-seed mean scaffold RMSE against a "
            "single-seed random-split RMSE. The ensembled scaffold RMSE is recorded for "
            "reference but is NOT used for optimism, because the random side is not "
            "ensembled and seed-ensembling helps the neural models far more than the forest."
        ),
        "note": (
            "Optimism is scaffold RMSE minus random RMSE: how much better a random-split "
            "evaluation would have looked on identical data and models. Each split selects "
            "its own epoch count on its own validation fold; reusing the scaffold-selected "
            "count on the random split under-trains the neural models there and understates "
            "their optimism."
        ),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP):
        control[label] = {
            "random_rmse": random_rmse[label],
            "scaffold_rmse": scaffold[label],
            "scaffold_rmse_ensembled": scaffold_ensembled[label],
            "optimism": scaffold[label] - random_rmse[label],
            "n_seeds_each_side": 1,
        }

    # Merge into diagnostics.json rather than overwriting it: the learning curve is
    # produced by 07_diagnostics.py and is unaffected by this control.
    out = RESULTS / "diagnostics.json"
    payload = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    payload["random_split_control"] = control
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"{'model':38s} {'random':>8s} {'scaffold':>9s} {'optimism':>9s}")
    for label in (LABEL_RF, LABEL_GNN, LABEL_CHEMPROP):
        block = control[label]
        print(f"{label:38s} {block['random_rmse']:8.4f} {block['scaffold_rmse']:9.4f} "
              f"{block['optimism']:+9.4f}")
    print()
    print(f"wrote random_split_control into diagnostics.json in "
          f"{control['runtime_seconds'] / 60:.0f} min")


if __name__ == "__main__":
    main()
