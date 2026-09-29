"""Does the split-dependence result hold up on other targets, or is it an HDAC1 thing?

On HDAC1 the split changes which model appears to win. On one target that's an anecdote,
so this re-runs the core experiment on three:

  * HDAC1  (CHEMBL325)  - the original, re-run under this reduced protocol so the
                          cross-target table is internally consistent
  * HDAC6  (CHEMBL1865) - same enzyme family, and HDAC1/HDAC6 selectivity is a real
                          med-chem problem, so this tests within-family replication
  * hERG   (CHEMBL240)  - a completely different target class, to see whether the effect
                          survives a change of chemistry

Everything reuses the same src/ modules as the main pipeline, so the cleaning, scaffold
assignment, featurisation and metrics are identical by construction rather than because I
reimplemented them the same way twice.

Reduced protocol, said plainly: three seeds instead of five, Chemprop only (GINE was in
the main study to show the result wasn't implementation-specific, and it did that), and no
forest hyperparameter search - the HDAC1 winner is reused. The question here is whether an
effect replicates in sign and rough size, not what each target's best RMSE is.
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
from src.chembl_fetch import fetch_activities, fetch_assays, fetch_target  # noqa: E402
from src.scaffold_split import build_split  # noqa: E402
from src.standardize import aggregate_by_key, compute_pic50, standardize_frame  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "multitarget.json"

TARGETS: tuple[tuple[str, str, str], ...] = (
    ("CHEMBL325", "HDAC1", "Histone deacetylase 1"),
    ("CHEMBL1865", "HDAC6", "Protein deacetylase HDAC6"),
    ("CHEMBL240", "hERG", "hERG potassium channel"),
)

SEEDS: tuple[int, ...] = (0, 1, 2)
RANDOM_SPLIT_SEED = 20260925
MIN_CONFIDENCE_SCORE = 9
EXPECTED_ORGANISM = "Homo sapiens"
MAX_PIC50_RANGE = 1.0
# Winning HDAC1 forest configuration, reused rather than re-searched per target.
RF_CONFIG = {"max_features": 0.1, "min_samples_leaf": 1, "n_estimators": 1000,
             "bootstrap": True, "n_jobs": -1}


def load_target(target_id: str) -> tuple[pd.DataFrame, dict]:
    """Fetch and clean one target through exactly the same path as scripts 01 and 02."""
    target = fetch_target(target_id)
    activities, _ = fetch_activities(
        target_chembl_id=target_id, standard_type="IC50", assay_type="B",
        standard_units="nM", standard_relation="=", require_pchembl=True,
    )
    assays, _ = fetch_assays(activities["assay_chembl_id"].dropna().unique().tolist())
    merged = activities.merge(
        assays[["assay_chembl_id", "confidence_score"]], on="assay_chembl_id", how="left"
    )
    merged["standard_value"] = merged["standard_value"].astype(float)

    funnel = {"fetched": len(merged)}
    step = merged[merged["target_organism"] == EXPECTED_ORGANISM]
    step = step[step["standard_value"] > 0]
    step = step[step["confidence_score"] == MIN_CONFIDENCE_SCORE]
    funnel["confidence_9_human"] = len(step)

    standardized, _ = standardize_frame(step, smiles_column="canonical_smiles")
    standardized["pIC50"] = compute_pic50(standardized["standard_value"])
    max_dev = float(
        (standardized["pIC50"] - standardized["pchembl_value"].astype(float)).abs().max()
    )
    if max_dev > 0.01:
        raise SystemExit(f"{target_id}: pIC50 disagrees with pchembl_value by {max_dev}")

    clean, counts = aggregate_by_key(
        standardized, key_column="inchikey", value_column="pIC50",
        max_range=MAX_PIC50_RANGE, carry_columns=("std_canonical_smiles", "molecule_chembl_id"),
    )
    clean = clean.rename(columns={"std_canonical_smiles": "canonical_smiles"})
    funnel |= counts
    meta = {
        "target_chembl_id": target_id,
        "pref_name": target.get("pref_name"),
        "organism": target.get("organism"),
        "funnel": funnel,
        "max_abs_pic50_minus_pchembl": max_dev,
        "n_compounds": len(clean),
    }
    return clean, meta


def make_random_split(frame: pd.DataFrame, sizes: dict[str, int], seed: int) -> pd.DataFrame:
    """Assign folds at random at the same fold sizes, as the control for the scaffold split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(frame))
    fold = np.empty(len(frame), dtype=object)
    fold[perm[: sizes["train"]]] = "train"
    fold[perm[sizes["train"] : sizes["train"] + sizes["val"]]] = "val"
    fold[perm[sizes["train"] + sizes["val"] :]] = "test"
    out = frame.copy().reset_index(drop=True)
    out["fold"] = fold
    return out


def score_split(frame: pd.DataFrame, label: str) -> dict:
    """Fit the forest and Chemprop on one split and return their test metrics."""
    train = frame[frame.fold == "train"].reset_index(drop=True)
    val = frame[frame.fold == "val"].reset_index(drop=True)
    test = frame[frame.fold == "test"].reset_index(drop=True)
    fit = pd.concat([train, val], ignore_index=True)
    y_fit = fit.pIC50.to_numpy(dtype=float)
    y_test = test.pIC50.to_numpy(dtype=float)

    X_fit, _ = F.build_features(fit.canonical_smiles.tolist())
    X_test, _ = F.build_features(test.canonical_smiles.tolist())
    res = rf.fit_predict_multiseed(
        X_fit.astype(np.float32), y_fit, X_test.astype(np.float32),
        RF_CONFIG, y_eval=y_test, seeds=SEEDS, verbose=False,
    )
    rf_rmse = ev.compute_metric("rmse", y_test, res.mean_prediction)
    rf_rho = ev.compute_metric("spearman", y_test, res.mean_prediction)

    # Epoch count chosen on the validation fold, then refit on train + val, as in the
    # main study. The test fold is scored once per model.
    _, _, _, best_epoch = CP.train_once(
        train, val, seed=0, max_epochs=CP.MAX_EPOCHS, use_early_stopping=True
    )
    cp_preds = []
    for seed in SEEDS:
        model, _, _, _ = CP.train_once(
            fit, None, seed=seed, max_epochs=best_epoch, use_early_stopping=False
        )
        cp_preds.append(CP.predict(model, test))
    cp_mean = np.vstack(cp_preds).mean(axis=0)
    cp_rmse = ev.compute_metric("rmse", y_test, cp_mean)
    cp_rho = ev.compute_metric("spearman", y_test, cp_mean)

    sims, _ = F.max_similarity_to_reference(
        test.canonical_smiles.tolist(), fit.canonical_smiles.tolist()
    )
    delta = ev.paired_bootstrap_delta(
        y_test, res.mean_prediction, cp_mean, "rmse",
        label_a="random forest", label_b="Chemprop",
    )
    print(f"        {label:9s} RF {rf_rmse:.4f}  Chemprop {cp_rmse:.4f}  "
          f"(delta {delta.delta:+.4f}, p={delta.p_value:.3f}, sig={delta.significant})  "
          f"median sim {np.median(sims):.3f}")
    return {
        "fold_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "random_forest": {"rmse": rf_rmse, "spearman": rf_rho},
        "chemprop": {"rmse": cp_rmse, "spearman": cp_rho, "selected_epochs": int(best_epoch)},
        "delta_rf_minus_chemprop": delta.to_dict(),
        "test_to_train_similarity": {
            "median": float(np.median(sims)),
            "fraction_ge_0.85": float((sims >= 0.85).mean()),
            "n_identical_fingerprint": int((sims >= 1.0).sum()),
        },
    }


def parse_args() -> argparse.Namespace:
    """Allow a subset of targets and a custom output file, so targets can run in parallel.

    The per-target work is independent, so on a multi-core machine the targets are better
    run as concurrent processes than sequentially. Each writes its own JSON, and the
    results are merged afterwards.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets", default="",
        help="comma-separated short names (e.g. hERG). Default: all.",
    )
    parser.add_argument(
        "--out", default="multitarget.json",
        help="output filename inside results/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUT_JSON, TARGETS
    OUT_JSON = RESULTS / args.out
    if args.targets:
        wanted = {name.strip().lower() for name in args.targets.split(",") if name.strip()}
        TARGETS = tuple(spec for spec in TARGETS if spec[1].lower() in wanted)
        if not TARGETS:
            raise SystemExit(f"no targets matched {sorted(wanted)}")
    print(f"targets: {[s[1] for s in TARGETS]}  ->  results/{args.out}", flush=True)

    t_start = time.time()
    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "models": ["random forest (ECFP4 + descriptors)", "Chemprop D-MPNN"],
            "rf_config": RF_CONFIG,
            "note": (
                "Reduced relative to the HDAC1 deep dive: 3 seeds not 5, Chemprop only, and "
                "the HDAC1 forest configuration reused rather than re-searched per target. "
                "Tests whether the split-dependence effect replicates, not each target's best "
                "achievable RMSE."
            ),
        },
        "targets": {},
    }

    for target_id, short, expected_name in TARGETS:
        print(f"\n=== {short} ({target_id}) ===", flush=True)
        t0 = time.time()
        clean, meta = load_target(target_id)
        print(f"      {meta['n_compounds']} compounds after cleaning "
              f"({meta['pref_name']})", flush=True)

        split_result = build_split(clean, smiles_column="canonical_smiles")
        scaffold_frame = split_result.frame
        sizes = {f: int((scaffold_frame.fold == f).sum()) for f in ("train", "val", "test")}
        random_frame = make_random_split(clean, sizes, RANDOM_SPLIT_SEED)

        scaffold = score_split(scaffold_frame, "scaffold")
        random_split = score_split(random_frame, "random")

        optimism = {
            model: scaffold[key]["rmse"] - random_split[key]["rmse"]
            for model, key in (("random_forest", "random_forest"), ("chemprop", "chemprop"))
        }
        ratio = (optimism["random_forest"] / optimism["chemprop"]
                 if optimism["chemprop"] else float("nan"))
        print(f"        optimism  RF {optimism['random_forest']:+.4f}   "
              f"Chemprop {optimism['chemprop']:+.4f}   ratio {ratio:.2f}x")

        payload["targets"][short] = {
            **meta,
            "n_scaffold_groups": int(split_result.frame.scaffold_group_id.nunique()),
            "scaffold_split": scaffold,
            "random_split": random_split,
            "optimism": optimism,
            "optimism_ratio_rf_over_chemprop": (
                float(ratio) if optimism["chemprop"] else None
            ),
            "seconds": round(time.time() - t0, 1),
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"      {short} done in {(time.time() - t0) / 60:.0f} min "
              f"(partial results written)", flush=True)

    payload["runtime_seconds"] = round(time.time() - t_start, 1)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY: scaffold-split optimism by target ===")
    print(f"{'target':8s} {'n':>6s}  {'RF scaf':>8s} {'RF rand':>8s} {'RF opt':>8s}  "
          f"{'CP scaf':>8s} {'CP rand':>8s} {'CP opt':>8s}  {'ratio':>6s}")
    for short, block in payload["targets"].items():
        s, r, o = block["scaffold_split"], block["random_split"], block["optimism"]
        ratio = block["optimism_ratio_rf_over_chemprop"]
        ratio_text = f"{ratio:6.2f}" if ratio is not None else "   n/a"
        print(f"{short:8s} {block['n_compounds']:6d}  "
              f"{s['random_forest']['rmse']:8.4f} {r['random_forest']['rmse']:8.4f} "
              f"{o['random_forest']:+8.4f}  "
              f"{s['chemprop']['rmse']:8.4f} {r['chemprop']['rmse']:8.4f} "
              f"{o['chemprop']:+8.4f}  {ratio_text}")
    print(f"\nwrote {OUT_JSON.name} in {payload['runtime_seconds'] / 60:.0f} min")


if __name__ == "__main__":
    main()
