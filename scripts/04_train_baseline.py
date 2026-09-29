"""The random-forest baseline. Tuned on val, scored once on test.

In order:

1. Load the split through the SHA-256-verified contract. This script never imports
   scaffold_split, so it can't recompute or reorder the folds.
2. Featurise train and val only.
3. Grid search on val.
4. Refit the winner on train + val across five seeds.
5. Read the test fold exactly once, and score.

The reference baselines (train mean, majority class, random, 1-NN Tanimoto) always get
emitted. Without them a model's numbers don't mean anything.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import baseline_rf as rf  # noqa: E402
from src import evaluate as ev  # noqa: E402
from src import featurize as F  # noqa: E402
from src.split_contract import load_split  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PREDICTIONS = RESULTS / "predictions_rf.npz"
METRICS_JSON = RESULTS / "metrics_rf.json"


def main() -> None:
    t_start = time.time()

    print("[1/7] Loading the verified split ...")
    bundle = load_split(DATA / "split_assignment.csv", RESULTS / "split_manifest.json")
    train, val = bundle.train, bundle.val
    print(f"      sha256={bundle.sha256[:16]}...  sizes={bundle.fold_sizes}")

    print("[2/7] Featurising train and val (test not yet read) ...")
    X_train, block = F.build_features(train.canonical_smiles.tolist())
    X_val, _ = F.build_features(val.canonical_smiles.tolist())
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_train = train.pIC50.to_numpy(dtype=np.float64)
    y_val = val.pIC50.to_numpy(dtype=np.float64)
    print(f"      X_train {X_train.shape}  X_val {X_val.shape}  "
          f"({block.n_fingerprint_bits} bits + {block.n_descriptors} descriptors)")
    if not np.isfinite(X_train).all() or not np.isfinite(X_val).all():
        raise SystemExit("non-finite feature values before fitting")

    print("[3/7] Grid search on the validation fold only ...")
    tuning = rf.tune_random_forest(X_train, y_train, X_val, y_val, seed=0)
    print(f"      best: {tuning.best_config}")
    print(f"      val RMSE {tuning.best_val_rmse:.4f}  val rho {tuning.best_val_spearman:.4f}")
    if bundle.n_test_accesses != 0:
        raise SystemExit(f"test fold read during tuning ({bundle.n_test_accesses} accesses)")

    print("[4/7] Refitting the winner on train + val across seeds ...")
    fit_frame_smiles = train.canonical_smiles.tolist() + val.canonical_smiles.tolist()
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    # The single permitted read of the test fold.
    test = bundle.test()
    if bundle.n_test_accesses != 1:
        raise SystemExit(f"test fold accessed {bundle.n_test_accesses} times, expected 1")
    X_test, _ = F.build_features(test.canonical_smiles.tolist())
    X_test = X_test.astype(np.float32)
    y_test = test.pIC50.to_numpy(dtype=np.float64)

    multiseed = rf.fit_predict_multiseed(
        X_fit, y_fit, X_test, tuning.best_config, y_eval=y_test, seeds=rf.SEEDS
    )
    y_pred = multiseed.mean_prediction
    print(f"      {multiseed.n_seeds} seeds in {multiseed.fit_seconds:.0f}s; "
          f"per-seed RMSE mean {np.mean(multiseed.per_seed_rmse):.4f} "
          f"sd {np.std(multiseed.per_seed_rmse, ddof=1):.4f}")

    print("[5/7] Reference baselines ...")
    train_fps = F.bitvects(fit_frame_smiles)
    test_fps = F.bitvects(test.canonical_smiles.tolist())
    nn_pred, nn_sim = ev.one_nn_tanimoto_predict(train_fps, y_fit, test_fps)
    references = {
        "mean_predictor": ev.mean_predictor(y_fit, len(y_test)),
        "majority_class": ev.majority_class_predictor(y_fit, len(y_test), ev.THRESHOLD_PRIMARY),
        "random": ev.random_predictor(len(y_test), float(y_fit.min()), float(y_fit.max())),
        "one_nn_tanimoto": nn_pred,
    }
    for name, pred in references.items():
        print(f"      {name:16s} RMSE {ev.compute_metric('rmse', y_test, pred):.4f}")

    print("[6/7] Feature-block ablations (single seed, footnote rows) ...")
    ablations: dict[str, dict] = {}
    ablation_specs = {
        "ecfp4_only": dict(include_descriptors=False, generator=None),
        "descriptors_only": dict(include_descriptors=True, generator=None, descriptors_only=True),
        "fcfp4_plus_desc": dict(include_descriptors=True, generator=F.make_morgan_generator(use_features=True)),
        "ecfp6_plus_desc": dict(include_descriptors=True, generator=F.make_morgan_generator(radius=3)),
    }
    for name, spec in ablation_specs.items():
        desc_only = spec.pop("descriptors_only", False)
        Xa_fit, _ = F.build_features(fit_frame_smiles, **spec)
        Xa_test, _ = F.build_features(test.canonical_smiles.tolist(), **spec)
        if desc_only:
            Xa_fit = Xa_fit[:, block.n_fingerprint_bits:]
            Xa_test = Xa_test[:, block.n_fingerprint_bits:]
        res = rf.fit_predict_multiseed(
            Xa_fit.astype(np.float32), y_fit, Xa_test.astype(np.float32),
            tuning.best_config, y_eval=y_test, seeds=(0,), verbose=False,
        )
        m = ev.regression_metrics(y_test, res.mean_prediction)
        ablations[name] = {"n_features": int(Xa_fit.shape[1]), **m.to_dict()}
        print(f"      {name:18s} n_feat={Xa_fit.shape[1]:5d}  RMSE {m.rmse:.4f}  rho {m.spearman:.4f}")

    print("[7/7] Scoring and writing artifacts ...")
    payload = {
        "model": "RandomForestRegressor (ECFP4 2048 + 12 descriptors)",
        "split_sha256": bundle.sha256,
        "evaluate_module_sha256": ev.module_sha256(),
        "n_seeds": multiseed.n_seeds,
        "refit_on_train_val": True,
        "n_test_accesses": bundle.n_test_accesses,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_fit": int(len(y_fit)),
        "n_test": int(len(y_test)),
        "feature_block": block.to_dict(),
        "tuning": tuning.to_dict(),
        "multiseed": multiseed.to_dict(),
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
        "references": {
            name: {
                "regression": ev.regression_metrics(y_test, pred).to_dict(),
                "classification": ev.classification_metrics(
                    y_test, pred, ev.THRESHOLD_PRIMARY
                ).to_dict(),
            }
            for name, pred in references.items()
        },
        "ablations": ablations,
        "importance": rf.split_importance(
            multiseed.feature_importances, block.n_fingerprint_bits, F.DESCRIPTOR_NAMES
        ),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    METRICS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        PREDICTIONS,
        y_test=y_test,
        y_pred=y_pred,
        per_seed=multiseed.per_seed_predictions,
        nn_similarity=nn_sim,
        inchikey=test.inchikey.to_numpy(dtype=str),
        **{f"ref_{k}": v for k, v in references.items()},
    )

    reg = payload["regression"]
    cls = payload["classification"][str(ev.THRESHOLD_PRIMARY)]
    print(f"\n      TEST  RMSE {reg['rmse']:.4f}  MAE {reg['mae']:.4f}  R2 {reg['r2']:.4f}  "
          f"rho {reg['spearman']:.4f}")
    print(f"      TEST  ROC-AUC {cls['roc_auc']:.4f}  PR-AUC {cls['pr_auc']:.4f}  "
          f"MCC {cls['mcc']:.4f}  (pos_rate {cls['pos_rate']:.4f})")
    print(f"      wrote {METRICS_JSON.name}, {PREDICTIONS.name}")
    print(f"04_train_baseline.py complete in {payload['runtime_seconds']:.0f}s.")


if __name__ == "__main__":
    main()
