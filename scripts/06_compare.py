"""Phase 6: the honest comparison, computed entirely from the two models' saved predictions.

Deliberately cheap and deterministic: it retrains nothing, so the headline table can be
regenerated in seconds and checked against the README. The expensive diagnostics that
need retraining (learning curve, random-split control) live in 07_diagnostics.py.

The headline statistic is the paired bootstrap on the difference, not the two marginal
intervals. Two overlapping confidence intervals do not mean two models are
indistinguishable, and reading them that way is the most common way a comparison like
this gets misreported.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import plotting as P  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"

LABEL_RF = "Random forest (ECFP4 + descriptors)"
LABEL_GNN = "GINE graph neural network"


def _load() -> tuple[dict, dict, np.lib.npyio.NpzFile, np.lib.npyio.NpzFile]:
    """Load both metrics files and prediction bundles, refusing any protocol mismatch."""
    rf_metrics = json.loads((RESULTS / "metrics_rf.json").read_text(encoding="utf-8"))
    gnn_metrics = json.loads((RESULTS / "metrics_gnn.json").read_text(encoding="utf-8"))
    rf_pred = np.load(RESULTS / "predictions_rf.npz", allow_pickle=False)
    gnn_pred = np.load(RESULTS / "predictions_gnn.npz", allow_pickle=False)

    # These three checks are the reason the comparison can be trusted at all.
    if rf_metrics["split_sha256"] != gnn_metrics["split_sha256"]:
        raise SystemExit(
            "SPLIT MISMATCH: the two models were scored on different splits.\n"
            f"  rf : {rf_metrics['split_sha256']}\n  gnn: {gnn_metrics['split_sha256']}"
        )
    if rf_metrics["evaluate_module_sha256"] != gnn_metrics["evaluate_module_sha256"]:
        raise SystemExit("EVALUATE MISMATCH: the two models were scored by different harness code.")
    if not np.array_equal(rf_pred["y_test"], gnn_pred["y_test"]):
        raise SystemExit("TEST LABEL MISMATCH: the two models saw different test labels.")
    for name, m in (("rf", rf_metrics), ("gnn", gnn_metrics)):
        if m["n_seeds"] != 5 or not m["refit_on_train_val"]:
            raise SystemExit(
                f"PROTOCOL ASYMMETRY in {name}: n_seeds={m['n_seeds']}, "
                f"refit_on_train_val={m['refit_on_train_val']}"
            )
        if m["n_test_accesses"] != 1:
            raise SystemExit(f"{name} read the test fold {m['n_test_accesses']} times, expected 1")
    return rf_metrics, gnn_metrics, rf_pred, gnn_pred


def main() -> None:
    print("[1/6] Loading model artifacts and verifying protocol symmetry ...")
    rf_metrics, gnn_metrics, rf_pred, gnn_pred = _load()
    y_test = rf_pred["y_test"]
    pred_rf = rf_pred["y_pred"]
    pred_gnn = gnn_pred["y_pred"]
    nn_sim = rf_pred["nn_similarity"]
    print(f"      split sha256 {rf_metrics['split_sha256'][:16]}... (identical for both)")
    print(f"      harness sha256 {rf_metrics['evaluate_module_sha256'][:16]}... (identical for both)")
    print(f"      n_test={len(y_test)}, both models 5 seeds, both refit on train+val")

    print("[2/6] Assembling the main table ...")
    models: dict[str, np.ndarray] = {
        "mean predictor": rf_pred["ref_mean_predictor"],
        "majority class": rf_pred["ref_majority_class"],
        "random scores": rf_pred["ref_random"],
        "1-NN Tanimoto": rf_pred["ref_one_nn_tanimoto"],
        LABEL_RF: pred_rf,
        LABEL_GNN: pred_gnn,
    }
    table_rows = []
    for name, pred in models.items():
        reg = ev.regression_metrics(y_test, pred)
        cls = ev.classification_metrics(y_test, pred, ev.THRESHOLD_PRIMARY)
        row = {"model": name, **reg.to_dict(), **{f"cls_{k}": v for k, v in cls.to_dict().items()}}
        if name in (LABEL_RF, LABEL_GNN):
            row["rmse_ci"] = ev.bootstrap_ci(y_test, pred, "rmse").to_dict()
            row["spearman_ci"] = ev.bootstrap_ci(y_test, pred, "spearman").to_dict()
            row["roc_auc_ci"] = ev.bootstrap_ci(
                y_test, pred, "roc_auc", ev.THRESHOLD_PRIMARY
            ).to_dict()
        table_rows.append(row)
        print(f"      {name:38s} RMSE {reg.rmse:.4f}  rho {reg.spearman:.4f}  "
              f"R2 {reg.r2:6.3f}  ROC-AUC {cls.roc_auc:.4f}")

    print("[3/6] Paired bootstrap on the RF - GNN difference (the headline) ...")
    deltas: dict[str, dict] = {}
    for metric in ev.METRIC_NAMES:
        threshold = ev.THRESHOLD_PRIMARY if metric in ev.CLASSIFICATION_METRICS else None
        d = ev.paired_bootstrap_delta(
            y_test, pred_rf, pred_gnn, metric, threshold,
            label_a=LABEL_RF, label_b=LABEL_GNN,
        )
        deltas[metric] = d.to_dict()
        flag = "*" if d.significant else " "
        print(f"      {flag} {metric:18s} delta {d.delta:+.4f}  95% CI [{d.lo:+.4f}, {d.hi:+.4f}]  "
              f"p={d.p_value:.3f}  favours={d.favours}")

    headline = ev.DeltaResult(**deltas["rmse"])
    verdict_text = ev.verdict(headline)
    print(f"\n      VERDICT: {verdict_text}")

    # Seed noise versus test-set noise: if seed sd exceeds the bootstrap half-width, the
    # model is under-determined at this data size, which is itself an answer to "why".
    rmse_ci = ev.bootstrap_ci(y_test, pred_gnn, "rmse")
    half_width = (rmse_ci.hi - rmse_ci.lo) / 2
    gnn_seed_sd = gnn_metrics["per_seed_rmse_sd"]
    rf_seed_sd = rf_metrics["multiseed"]["per_seed_rmse_sd"]
    print(f"      GNN seed sd {gnn_seed_sd:.4f} vs bootstrap half-width {half_width:.4f} "
          f"-> {'seed noise dominates' if gnn_seed_sd > half_width else 'test-set noise dominates'}")

    print("[4/6] Similarity-stratified and replicate-stratified analysis ...")
    bin_idx, bin_labels = P.bin_by_similarity(nn_sim)
    split = pd.read_csv(DATA / "split_assignment.csv")
    test_meta = split[split.fold == "test"].reset_index(drop=True)

    stratified: dict[str, list] = {"bins": bin_labels, "counts": [], LABEL_RF: [], LABEL_GNN: []}
    strat_ci: dict[str, tuple[list, list]] = {LABEL_RF: ([], []), LABEL_GNN: ([], [])}
    for b in range(len(bin_labels)):
        mask = bin_idx == b
        stratified["counts"].append(int(mask.sum()))
        for label, pred in ((LABEL_RF, pred_rf), (LABEL_GNN, pred_gnn)):
            if mask.sum() >= 5:
                value = ev.compute_metric("rmse", y_test[mask], pred[mask])
                ci = ev.bootstrap_ci(y_test[mask], pred[mask], "rmse")
                lo, hi = ci.lo, ci.hi
            else:
                value = lo = hi = float("nan")
            stratified[label].append(value)
            strat_ci[label][0].append(lo)
            strat_ci[label][1].append(hi)
        print(f"      {bin_labels[b]:12s} n={stratified['counts'][b]:4d}  "
              f"RF {stratified[LABEL_RF][b]:.4f}  GNN {stratified[LABEL_GNN][b]:.4f}")

    replicate_strata = {}
    for name, mask in (
        ("single measurement", test_meta.n_measurements.to_numpy() == 1),
        ("replicated (>=2)", test_meta.n_measurements.to_numpy() >= 2),
    ):
        if mask.sum() >= 5:
            replicate_strata[name] = {
                "n": int(mask.sum()),
                LABEL_RF: ev.compute_metric("rmse", y_test[mask], pred_rf[mask]),
                LABEL_GNN: ev.compute_metric("rmse", y_test[mask], pred_gnn[mask]),
            }
            r = replicate_strata[name]
            print(f"      {name:20s} n={r['n']:4d}  RF {r[LABEL_RF]:.4f}  GNN {r[LABEL_GNN]:.4f}")

    res_rf = y_test - pred_rf
    res_gnn = y_test - pred_gnn
    residual_r = float(np.corrcoef(res_rf, res_gnn)[0, 1])
    print(f"      residual correlation RF vs GNN: r = {residual_r:.3f}")

    print("[5/6] Figures ...")
    figs = {}
    figs["fig_parity.png"] = P.save_figure(
        P.parity_panels(
            y_test,
            [P.Series(LABEL_RF, pred_rf, P.MODEL_COLOURS["rf"]),
             P.Series(LABEL_GNN, pred_gnn, P.MODEL_COLOURS["gnn"])],
            colour_by=nn_sim,
        ),
        RESULTS / "fig_parity.png",
    )
    figs["fig_stratified_by_similarity.png"] = P.save_figure(
        P.stratified_bars(
            bin_labels,
            {LABEL_RF: stratified[LABEL_RF], LABEL_GNN: stratified[LABEL_GNN]},
            errors=strat_ci, counts=stratified["counts"],
            title="Test error by structural distance from the training set",
        ),
        RESULTS / "fig_stratified_by_similarity.png",
    )
    figs["fig_error_distributions.png"] = P.save_figure(
        P.error_distributions({LABEL_RF: res_rf, LABEL_GNN: res_gnn}, pair=(LABEL_RF, LABEL_GNN)),
        RESULTS / "fig_error_distributions.png",
    )
    figs["fig_split_diagnostics.png"] = P.save_figure(
        P.split_diagnostics(
            ["train", "val", "test"],
            {f: split[split.fold == f].pIC50.to_numpy() for f in ("train", "val", "test")},
            split.groupby("scaffold_group_id").size().to_numpy(),
            nn_sim,
        ),
        RESULTS / "fig_split_diagnostics.png",
    )
    imp = rf_metrics["importance"]
    figs["fig_rf_importance.png"] = P.save_figure(
        P.importance_split(
            imp["fingerprint_total"], imp["descriptor_total"],
            dict(sorted(imp["descriptors"].items(), key=lambda kv: -kv[1])[:8]),
        ),
        RESULTS / "fig_rf_importance.png",
    )
    for name in figs:
        print(f"      wrote {name}")

    print("[6/6] Writing comparison.json ...")
    payload = {
        "split_sha256": rf_metrics["split_sha256"],
        "evaluate_module_sha256": rf_metrics["evaluate_module_sha256"],
        "n_test": int(len(y_test)),
        "primary_threshold": ev.THRESHOLD_PRIMARY,
        "labels": {"a": LABEL_RF, "b": LABEL_GNN},
        "table": table_rows,
        "deltas": deltas,
        "verdict": verdict_text,
        "verdict_metric": "rmse",
        "seed_vs_bootstrap_noise": {
            "gnn_seed_sd": gnn_seed_sd,
            "rf_seed_sd": rf_seed_sd,
            "bootstrap_half_width_gnn_rmse": half_width,
            "seed_noise_dominates": bool(gnn_seed_sd > half_width),
        },
        "stratified_by_similarity": {
            "bins": bin_labels, "counts": stratified["counts"],
            LABEL_RF: stratified[LABEL_RF], LABEL_GNN: stratified[LABEL_GNN],
        },
        "stratified_by_replicates": replicate_strata,
        "residual_correlation": residual_r,
        "figures": [str(p.name) for p in figs.values()],
    }
    (RESULTS / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("      wrote comparison.json")
    print("06_compare.py complete.")


if __name__ == "__main__":
    main()
