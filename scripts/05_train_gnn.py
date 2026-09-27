"""Phase 5: the GINE message-passing network, on the same split and the same harness.

Deliberately sequential after Phase 4, and deliberately symmetric with it:

* the split arrives through the same SHA-256-verified contract, and this script never
  imports ``scaffold_split`` either;
* tuning happens on the validation fold only;
* the winner is refit on train + val, exactly as the forest is, for the epoch count that
  early stopping selected during tuning;
* five seeds, reported as mean and sd;
* every metric comes from ``src.evaluate``.

Running one model over five seeds and the other over one, or refitting one on train+val
and not the other, would quietly decide the comparison before any data was seen.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import gnn as G  # noqa: E402
from src.split_contract import load_split  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PREDICTIONS = RESULTS / "predictions_gnn.npz"
METRICS_JSON = RESULTS / "metrics_gnn.json"

SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

# Small, honest search. The point is not to squeeze the GNN but to rule out the
# objection that it lost because nobody tuned it.
TUNING_GRID: tuple[dict, ...] = tuple(
    {"hidden": h, "depth": d, "dropout": p}
    for h in (64, 128)
    for d in (3, 4)
    for p in (0.0, 0.2)
)


def main() -> None:
    t_start = time.time()
    torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))

    print("[1/6] Loading the verified split ...")
    bundle = load_split(DATA / "split_assignment.csv", RESULTS / "split_manifest.json")
    train, val = bundle.train, bundle.val
    print(f"      sha256={bundle.sha256[:16]}...  sizes={bundle.fold_sizes}")

    print("[2/6] Building molecular graphs for train and val ...")
    train_graphs, train_stats = G.smiles_to_graphs(
        train.canonical_smiles.tolist(), train.pIC50.to_numpy()
    )
    val_graphs, val_stats = G.smiles_to_graphs(
        val.canonical_smiles.tolist(), val.pIC50.to_numpy()
    )
    print(f"      train {train_stats['n_graphs']} graphs, val {val_stats['n_graphs']} graphs")
    print(f"      atom dim {G.ATOM_FEATURE_DIM}, bond dim {G.BOND_FEATURE_DIM}, "
          f"OOV atom fraction {train_stats['oov_fraction']:.5f}")
    if train_stats["oov_fraction"] > 0.001:
        raise SystemExit(f"out-of-vocabulary atom fraction too high: {train_stats['oov_fraction']}")

    print("[3/6] Tuning on the validation fold only ...")
    rows = []
    for i, params in enumerate(TUNING_GRID, 1):
        cfg = G.GNNConfig(**params)
        t0 = time.time()
        _, res = G.train_model(train_graphs, val_graphs, cfg, seed=0)
        elapsed = time.time() - t0
        rows.append({
            "config": params, "val_rmse": res.best_val_rmse,
            "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
            "n_parameters": res.n_parameters, "seconds": round(elapsed, 1),
        })
        print(f"      [{i}/{len(TUNING_GRID)}] hidden={params['hidden']:3d} depth={params['depth']} "
              f"dropout={params['dropout']:.1f}  val RMSE {res.best_val_rmse:.4f}  "
              f"best_epoch {res.best_epoch:3d}  ({elapsed:.0f}s)")
    best = min(rows, key=lambda r: r["val_rmse"])
    best_cfg = G.GNNConfig(**best["config"])
    print(f"      best: {best['config']}  val RMSE {best['val_rmse']:.4f}  "
          f"best_epoch {best['best_epoch']}")
    if bundle.n_test_accesses != 0:
        raise SystemExit(f"test fold read during tuning ({bundle.n_test_accesses} accesses)")

    print("[4/6] Refitting on train + val across seeds ...")
    fit_graphs = train_graphs + val_graphs
    test = bundle.test()  # the single permitted read
    if bundle.n_test_accesses != 1:
        raise SystemExit(f"test fold accessed {bundle.n_test_accesses} times, expected 1")
    test_graphs, test_stats = G.smiles_to_graphs(
        test.canonical_smiles.tolist(), test.pIC50.to_numpy()
    )
    y_test = test.pIC50.to_numpy(dtype=np.float64)

    per_seed_preds, per_seed_rmse, seed_reports = [], [], []
    for seed in SEEDS:
        t0 = time.time()
        model, res = G.train_model(
            fit_graphs, None, best_cfg, seed=seed, fixed_epochs=int(best["best_epoch"])
        )
        pred = G.predict(model, test_graphs, res)
        rmse = ev.compute_metric("rmse", y_test, pred)
        per_seed_preds.append(pred)
        per_seed_rmse.append(rmse)
        seed_reports.append({
            "seed": seed, "epochs_run": res.epochs_run, "rmse": rmse,
            "n_parameters": res.n_parameters, "seconds": round(time.time() - t0, 1),
        })
        print(f"      seed {seed}: RMSE {rmse:.4f}  ({time.time() - t0:.0f}s)")

    per_seed = np.vstack(per_seed_preds)
    y_pred = per_seed.mean(axis=0)
    sd_seeds = float(np.std(per_seed_rmse, ddof=1))
    print(f"      per-seed RMSE mean {np.mean(per_seed_rmse):.4f} sd {sd_seeds:.4f}; "
          f"ensembled-mean RMSE {ev.compute_metric('rmse', y_test, y_pred):.4f}")

    print("[5/6] Scoring through the shared harness ...")
    payload = {
        "model": f"GINEConv x{best_cfg.depth}, hidden {best_cfg.hidden}, mean+sum readout",
        "architecture_note": (
            "GINEConv, a member of the Gilmer et al. (2017) message-passing family, chosen "
            "over a literal NNConv edge-network MPNN for parameter efficiency at this "
            "dataset size. Not a verbatim reimplementation of Gilmer 2017."
        ),
        "split_sha256": bundle.sha256,
        "evaluate_module_sha256": ev.module_sha256(),
        "n_seeds": len(SEEDS),
        "refit_on_train_val": True,
        "n_test_accesses": bundle.n_test_accesses,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_fit": int(len(fit_graphs)),
        "n_test": int(len(y_test)),
        "atom_feature_dim": G.ATOM_FEATURE_DIM,
        "bond_feature_dim": G.BOND_FEATURE_DIM,
        "graph_stats": {"train": train_stats, "val": val_stats, "test": test_stats},
        "config": best_cfg.to_dict(),
        "tuning": {"grid": rows, "best": best, "tuned_on": "validation fold only"},
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

    print("[6/6] Done.")
    reg = payload["regression"]
    cls = payload["classification"][str(ev.THRESHOLD_PRIMARY)]
    print(f"\n      TEST  RMSE {reg['rmse']:.4f}  MAE {reg['mae']:.4f}  R2 {reg['r2']:.4f}  "
          f"rho {reg['spearman']:.4f}")
    print(f"      TEST  ROC-AUC {cls['roc_auc']:.4f}  PR-AUC {cls['pr_auc']:.4f}  MCC {cls['mcc']:.4f}")
    print(f"      seed sd {sd_seeds:.4f} | parameters {seed_reports[0]['n_parameters']:,} "
          f"vs {payload['n_fit']:,} training molecules")
    print(f"05_train_gnn.py complete in {payload['runtime_seconds']:.0f}s.")


if __name__ == "__main__":
    main()
