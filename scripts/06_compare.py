"""Phase 6: the honest comparison, computed entirely from the models' saved predictions.

Deliberately cheap and deterministic: it retrains nothing, so the headline table can be
regenerated in seconds and checked against the README. The expensive diagnostics that
need retraining (learning curve, random-split control) live in 07_diagnostics.py.

Three models are compared where available - the fingerprint random forest, a hand-rolled
GINE network, and Chemprop's D-MPNN at author defaults. Chemprop is the field-standard
graph baseline, so the headline comparison is the forest against it; the GINE network is
reported alongside to show that the result is not an artifact of one particular
implementation.

The headline statistic is the paired bootstrap on the difference, not the marginal
intervals. Two overlapping confidence intervals do not mean two models are
indistinguishable, and reading them that way is the most common way a comparison like
this gets misreported.
"""

from __future__ import annotations

import itertools
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
LABEL_GINE = "GINE graph neural network"
LABEL_CHEMPROP = "Chemprop D-MPNN"

# (label, metrics file, predictions file, required)
MODEL_SPECS = [
    (LABEL_RF, "metrics_rf.json", "predictions_rf.npz", True),
    (LABEL_GINE, "metrics_gnn.json", "predictions_gnn.npz", True),
    (LABEL_CHEMPROP, "metrics_chemprop.json", "predictions_chemprop.npz", False),
]

# The forest against the field-standard graph model is the comparison that carries the
# claim; the others are reported so the reader can see it is not implementation-specific.
HEADLINE_PAIR = (LABEL_RF, LABEL_CHEMPROP)
FALLBACK_PAIR = (LABEL_RF, LABEL_GINE)


def _load() -> tuple[dict[str, dict], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load every available model, refusing any protocol mismatch between them."""
    metrics: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    bundles: dict[str, np.lib.npyio.NpzFile] = {}

    for label, mfile, pfile, required in MODEL_SPECS:
        mpath, ppath = RESULTS / mfile, RESULTS / pfile
        if not mpath.exists() or not ppath.exists():
            if required:
                raise SystemExit(f"missing required artifacts for {label}: {mfile} / {pfile}")
            print(f"      (skipping {label}: artifacts not present)")
            continue
        metrics[label] = json.loads(mpath.read_text(encoding="utf-8"))
        bundles[label] = np.load(ppath, allow_pickle=False)
        preds[label] = bundles[label]["y_pred"]

    labels = list(metrics)
    ref = labels[0]
    y_test = bundles[ref]["y_test"]

    # These checks are the reason the comparison can be trusted at all.
    for label in labels[1:]:
        if metrics[label]["split_sha256"] != metrics[ref]["split_sha256"]:
            raise SystemExit(
                f"SPLIT MISMATCH between {ref} and {label}:\n"
                f"  {ref}: {metrics[ref]['split_sha256']}\n  {label}: {metrics[label]['split_sha256']}"
            )
        if metrics[label]["evaluate_module_sha256"] != metrics[ref]["evaluate_module_sha256"]:
            raise SystemExit(f"EVALUATE MISMATCH: {label} was scored by different harness code.")
        if not np.array_equal(bundles[label]["y_test"], y_test):
            raise SystemExit(f"TEST LABEL MISMATCH: {label} saw different test labels.")
    for label in labels:
        m = metrics[label]
        if m["n_seeds"] != 5 or not m["refit_on_train_val"]:
            raise SystemExit(
                f"PROTOCOL ASYMMETRY in {label}: n_seeds={m['n_seeds']}, "
                f"refit_on_train_val={m['refit_on_train_val']}"
            )
        if m["n_test_accesses"] != 1:
            raise SystemExit(f"{label} read the test fold {m['n_test_accesses']} times, expected 1")

    return metrics, preds, y_test, bundles[LABEL_RF]["nn_similarity"]


def main() -> None:
    print("[1/6] Loading model artifacts and verifying protocol symmetry ...")
    metrics, preds, y_test, nn_sim = _load()
    rf_bundle = np.load(RESULTS / "predictions_rf.npz", allow_pickle=False)
    ref = next(iter(metrics))
    print(f"      split sha256 {metrics[ref]['split_sha256'][:16]}... (identical across models)")
    print(f"      harness sha256 {metrics[ref]['evaluate_module_sha256'][:16]}... (identical)")
    print(f"      n_test={len(y_test)}, models: {', '.join(metrics)}")

    print("[2/6] Assembling the main table ...")
    references = {
        "mean predictor": rf_bundle["ref_mean_predictor"],
        "majority class": rf_bundle["ref_majority_class"],
        "random scores": rf_bundle["ref_random"],
        "1-NN Tanimoto": rf_bundle["ref_one_nn_tanimoto"],
    }
    table_rows = []
    for name, pred in {**references, **preds}.items():
        reg = ev.regression_metrics(y_test, pred)
        cls = ev.classification_metrics(y_test, pred, ev.THRESHOLD_PRIMARY)
        row = {"model": name, "is_model": name in preds,
               **reg.to_dict(), **{f"cls_{k}": v for k, v in cls.to_dict().items()}}
        if name in preds:
            for metric in ("rmse", "spearman"):
                row[f"{metric}_ci"] = ev.bootstrap_ci(y_test, pred, metric).to_dict()
            row["roc_auc_ci"] = ev.bootstrap_ci(
                y_test, pred, "roc_auc", ev.THRESHOLD_PRIMARY
            ).to_dict()
            m = metrics[name]
            row["per_seed_rmse_mean"] = m.get("per_seed_rmse_mean") or m["multiseed"]["per_seed_rmse_mean"]
            row["per_seed_rmse_sd"] = m.get("per_seed_rmse_sd") or m["multiseed"]["per_seed_rmse_sd"]
        table_rows.append(row)
        print(f"      {name:38s} RMSE {reg.rmse:.4f}  rho {reg.spearman:.4f}  "
              f"R2 {reg.r2:6.3f}  ROC-AUC {cls.roc_auc:.4f}")

    print("[3/6] Paired bootstrap on every model pair ...")
    pairwise: dict[str, dict[str, dict]] = {}
    for a, b in itertools.combinations(preds, 2):
        key = f"{a} vs {b}"
        pairwise[key] = {}
        print(f"      --- {key} ---")
        for metric in ev.METRIC_NAMES:
            threshold = ev.THRESHOLD_PRIMARY if metric in ev.CLASSIFICATION_METRICS else None
            d = ev.paired_bootstrap_delta(
                y_test, preds[a], preds[b], metric, threshold, label_a=a, label_b=b
            )
            pairwise[key][metric] = d.to_dict()
            flag = "*" if d.significant else " "
            print(f"        {flag} {metric:18s} delta {d.delta:+.4f}  "
                  f"95% CI [{d.lo:+.4f}, {d.hi:+.4f}]  p={d.p_value:.3f}")

    headline_key = " vs ".join(HEADLINE_PAIR)
    if headline_key not in pairwise:
        headline_key = " vs ".join(FALLBACK_PAIR)
    headline = ev.DeltaResult(**pairwise[headline_key]["rmse"])
    verdict_text = ev.verdict(headline)
    print(f"\n      HEADLINE ({headline_key})")
    print(f"      VERDICT: {verdict_text}")

    any_significant = [
        f"{key} / {metric}"
        for key, block in pairwise.items()
        for metric, d in block.items()
        if d["significant"]
    ]
    print(f"      significant differences across all pairs and metrics: "
          f"{len(any_significant)} of {sum(len(b) for b in pairwise.values())}")
    for s in any_significant:
        print(f"        * {s}")

    print("[4/6] Similarity-stratified and replicate-stratified analysis ...")
    bin_idx, bin_labels = P.bin_by_similarity(nn_sim)
    split = pd.read_csv(DATA / "split_assignment.csv")
    test_meta = split[split.fold == "test"].reset_index(drop=True)

    stratified: dict[str, list] = {"bins": bin_labels, "counts": []}
    strat_ci: dict[str, tuple[list, list]] = {}
    for label in preds:
        stratified[label] = []
        strat_ci[label] = ([], [])
    for b in range(len(bin_labels)):
        mask = bin_idx == b
        stratified["counts"].append(int(mask.sum()))
        for label, pred in preds.items():
            if mask.sum() >= 5:
                value = ev.compute_metric("rmse", y_test[mask], pred[mask])
                ci = ev.bootstrap_ci(y_test[mask], pred[mask], "rmse")
                lo, hi = ci.lo, ci.hi
            else:
                value = lo = hi = float("nan")
            stratified[label].append(value)
            strat_ci[label][0].append(lo)
            strat_ci[label][1].append(hi)
        cells = "  ".join(f"{label.split()[0]} {stratified[label][b]:.4f}" for label in preds)
        print(f"      {bin_labels[b]:12s} n={stratified['counts'][b]:4d}  {cells}")

    replicate_strata = {}
    for name, mask in (
        ("single measurement", test_meta.n_measurements.to_numpy() == 1),
        ("replicated (>=2)", test_meta.n_measurements.to_numpy() >= 2),
    ):
        if mask.sum() >= 5:
            replicate_strata[name] = {"n": int(mask.sum())} | {
                label: ev.compute_metric("rmse", y_test[mask], pred[mask])
                for label, pred in preds.items()
            }
            cells = "  ".join(f"{k.split()[0]} {v:.4f}" for k, v in replicate_strata[name].items()
                              if k != "n")
            print(f"      {name:20s} n={replicate_strata[name]['n']:4d}  {cells}")

    residuals = {label: y_test - pred for label, pred in preds.items()}
    residual_corr = {
        f"{a} vs {b}": float(np.corrcoef(residuals[a], residuals[b])[0, 1])
        for a, b in itertools.combinations(preds, 2)
    }
    for key, value in residual_corr.items():
        print(f"      residual correlation {key}: r = {value:.3f}")

    print("[5/6] Figures ...")
    series = [P.Series(label, pred, P.resolve_colour(label) or "#555555")
              for label, pred in preds.items()]
    figs = {}
    figs["fig_parity.png"] = P.save_figure(
        P.parity_panels(y_test, series, colour_by=nn_sim), RESULTS / "fig_parity.png"
    )
    figs["fig_stratified_by_similarity.png"] = P.save_figure(
        P.stratified_bars(
            bin_labels, {label: stratified[label] for label in preds},
            errors=strat_ci, counts=stratified["counts"],
            title="Test error by structural distance from the training set",
        ),
        RESULTS / "fig_stratified_by_similarity.png",
    )
    figs["fig_error_distributions.png"] = P.save_figure(
        P.error_distributions(residuals, pair=HEADLINE_PAIR if LABEL_CHEMPROP in preds else FALLBACK_PAIR),
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
    imp = metrics[LABEL_RF]["importance"]
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
        "split_sha256": metrics[ref]["split_sha256"],
        "evaluate_module_sha256": metrics[ref]["evaluate_module_sha256"],
        "n_test": int(len(y_test)),
        "primary_threshold": ev.THRESHOLD_PRIMARY,
        "models": list(preds),
        "headline_pair": headline_key,
        "table": table_rows,
        "pairwise_deltas": pairwise,
        "deltas": pairwise[headline_key],
        "verdict": verdict_text,
        "verdict_metric": "rmse",
        "n_significant_differences": len(any_significant),
        "significant_differences": any_significant,
        "seed_noise": {
            label: {
                "per_seed_rmse_sd": (metrics[label].get("per_seed_rmse_sd")
                                     or metrics[label]["multiseed"]["per_seed_rmse_sd"]),
                "bootstrap_half_width_rmse": (
                    ev.bootstrap_ci(y_test, preds[label], "rmse").hi
                    - ev.bootstrap_ci(y_test, preds[label], "rmse").lo
                ) / 2,
            }
            for label in preds
        },
        "stratified_by_similarity": {
            "bins": bin_labels, "counts": stratified["counts"],
            **{label: stratified[label] for label in preds},
        },
        "stratified_by_replicates": replicate_strata,
        "residual_correlation": residual_corr,
        "figures": [str(p.name) for p in figs.values()],
    }
    (RESULTS / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("      wrote comparison.json")
    print("06_compare.py complete.")


if __name__ == "__main__":
    main()
