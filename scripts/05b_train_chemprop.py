"""Phase 5b: Chemprop's D-MPNN — the field-standard graph baseline — on the same split.

Why this script exists. The original comparison pitted a hand-rolled GINE network against the
random forest, which leaves an obvious objection open: that the GNN lost because it was a weak
GNN, not because graph models struggle at this data size. Chemprop's directed message-passing
network (Yang et al., J Chem Inf Model 2019) is the reference implementation the field actually
benchmarks against, so running it closes that objection with the standard tool rather than
with an argument.

Protocol is identical to 05_train_gnn.py, deliberately:
  * the split arrives through the same SHA-256-verified contract;
  * the epoch count is selected on the validation fold only;
  * the model is refit on train + val, as the forest and the GINE network both are;
  * five seeds, mean prediction;
  * every metric comes from src/evaluate.py.

One deliberate asymmetry, stated rather than hidden: no hyperparameter search is run here. The
Chemprop defaults are the authors' recommended configuration and are what the field cites, so
using them unmodified is the fairest available reference point, and searching over them would
cost hours of CPU for a baseline whose purpose is to be standard rather than optimal.
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src.chemprop_model import (  # noqa: E402
    BATCH_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    build_model,
    predict,
    train_once,
)
from src.split_contract import load_split  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PREDICTIONS = RESULTS / "predictions_chemprop.npz"
METRICS_JSON = RESULTS / "metrics_chemprop.json"

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def main() -> None:
    t_start = time.time()

    print("[1/5] Loading the verified split ...")
    bundle = load_split(DATA / "split_assignment.csv", RESULTS / "split_manifest.json")
    train, val = bundle.train, bundle.val
    print(f"      sha256={bundle.sha256[:16]}...  sizes={bundle.fold_sizes}")

    print("[2/5] Selecting the epoch count on the validation fold ...")
    t0 = time.time()
    sel_model, _, val_curve, best_epoch = train_once(
        train, val, seed=0, max_epochs=MAX_EPOCHS, use_early_stopping=True
    )
    val_pred = predict(sel_model, val)
    val_rmse = ev.compute_metric("rmse", val.pIC50.to_numpy(dtype=float), val_pred)
    print(f"      best_epoch {best_epoch} of {len(val_curve)} run; val RMSE {val_rmse:.4f} "
          f"({time.time() - t0:.0f}s)")
    if bundle.n_test_accesses != 0:
        raise SystemExit(f"test fold read during selection ({bundle.n_test_accesses} accesses)")

    print("[3/5] Refitting on train + val across seeds ...")
    fit_frame = pd.concat([train, val], ignore_index=True)
    test = bundle.test()  # the single permitted read
    if bundle.n_test_accesses != 1:
        raise SystemExit(f"test fold accessed {bundle.n_test_accesses} times, expected 1")
    y_test = test.pIC50.to_numpy(dtype=np.float64)

    per_seed_preds, per_seed_rmse, seed_reports = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, _, _, _ = train_once(
            fit_frame, None, seed=seed, max_epochs=best_epoch, use_early_stopping=False
        )
        pred = predict(model, test)
        rmse = ev.compute_metric("rmse", y_test, pred)
        per_seed_preds.append(pred)
        per_seed_rmse.append(rmse)
        seed_reports.append({
            "seed": seed, "epochs_run": best_epoch, "rmse": rmse,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"      seed {seed}: RMSE {rmse:.4f}  ({time.time() - t0:.0f}s)")

    per_seed = np.vstack(per_seed_preds)
    y_pred = per_seed.mean(axis=0)
    sd_seeds = float(np.std(per_seed_rmse, ddof=1))
    print(f"      per-seed RMSE mean {np.mean(per_seed_rmse):.4f} sd {sd_seeds:.4f}; "
          f"ensembled-mean RMSE {ev.compute_metric('rmse', y_test, y_pred):.4f}")

    print("[4/5] Scoring through the shared harness ...")
    n_params = sum(p.numel() for p in build_model().parameters() if p.requires_grad)
    payload = {
        "model": "Chemprop D-MPNN (defaults: bond message passing, mean aggregation)",
        "architecture_note": (
            "Chemprop 2.3.1 at author-default hyperparameters. No hyperparameter search was "
            "run: the defaults are the configuration the field benchmarks against, which is "
            "what makes this a reference point rather than a tuned competitor."
        ),
        "reference": "Yang et al., J Chem Inf Model 2019, 59(8):3370-3388",
        "split_sha256": bundle.sha256,
        "evaluate_module_sha256": ev.module_sha256(),
        "n_seeds": len(SEEDS),
        "refit_on_train_val": True,
        "n_test_accesses": bundle.n_test_accesses,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_fit": int(len(fit_frame)),
        "n_test": int(len(y_test)),
        "n_parameters": int(n_params),
        "config": {
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
            "selected_epochs": best_epoch, "hyperparameter_search": "none (author defaults)",
        },
        "selection": {
            "val_rmse": val_rmse, "val_loss_curve": val_curve,
            "tuned_on": "validation fold only (epoch count only)",
        },
        "seeds": seed_reports,
        "per_seed_rmse": per_seed_rmse,
        "per_seed_rmse_mean": float(np.mean(per_seed_rmse)),
        "per_seed_rmse_sd": sd_seeds,
        "regression": ev.regression_metrics(y_test, y_pred).to_dict(),
        "classification": {
            str(t): ev.classification_metrics(y_test, y_pred, t).to_dict() for t in ev.THRESHOLDS
        },
        "bootstrap": {
            m: ev.bootstrap_ci(
                y_test, y_pred, m, ev.THRESHOLD_PRIMARY if m in ev.CLASSIFICATION_METRICS else None
            ).to_dict()
            for m in ev.METRIC_NAMES
        },
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        PREDICTIONS, y_test=y_test, y_pred=y_pred, per_seed=per_seed,
        inchikey=test.inchikey.to_numpy(dtype=str),
    )

    print("[5/5] Done.")
    reg = payload["regression"]
    cls = payload["classification"][str(ev.THRESHOLD_PRIMARY)]
    print(f"\n      TEST  RMSE {reg['rmse']:.4f}  MAE {reg['mae']:.4f}  R2 {reg['r2']:.4f}  "
          f"rho {reg['spearman']:.4f}")
    print(f"      TEST  ROC-AUC {cls['roc_auc']:.4f}  PR-AUC {cls['pr_auc']:.4f}  MCC {cls['mcc']:.4f}")
    print(f"      seed sd {sd_seeds:.4f} | parameters {n_params:,}")
    print(f"05b_train_chemprop.py complete in {payload['runtime_seconds']:.0f}s.")


if __name__ == "__main__":
    main()
