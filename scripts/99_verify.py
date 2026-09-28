"""The executable proof. Run this before trusting any number in this repository.

It re-derives, from the committed artifacts alone, every claim the README makes:

* the split artifact still hashes to what the manifest recorded;
* the folds are disjoint by scaffold, structure and identifier;
* the two models consumed the identical split and the identical scoring code;
* neither model read the test fold more than once, and neither tuned on it;
* the protocol was symmetric (same seed count, both refit on train + val);
* the evaluation harness itself behaves correctly on synthetic inputs where the
  right answer is known in advance;
* every number quoted in the README matches the JSON it came from.

Exit code 0 means all of that held. Anything else means it did not.

Deliberately re-derives rather than re-reads: a verification pass that trusted the same
summary the README was written from would prove nothing.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev  # noqa: E402
from src import gates as gates_mod  # noqa: E402
from src.split_contract import load_split, sha256_file  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
SCRIPTS = ROOT / "scripts"

failures: list[str] = []
flags: list[str] = []


def check(name: str, passed: bool, observed: str = "", severity: str = "fail") -> None:
    """Record and print one verification result."""
    mark = "PASS" if passed else ("FAIL" if severity == "fail" else "FLAG")
    print(f"  {mark}  {name}" + (f"   {observed}" if observed else ""))
    if not passed:
        (failures if severity == "fail" else flags).append(f"{name}: {observed}")


# ---------------------------------------------------------------------------------
# 1. Harness self-test: does src/evaluate.py give known answers on known inputs?
# ---------------------------------------------------------------------------------


def verify_harness() -> None:
    """Check the metric code against cases whose correct answers are known a priori."""
    print("\n[1] Evaluation harness self-test")
    rng = np.random.default_rng(7)
    n = 400
    y = rng.normal(6.6, 1.0, n).clip(4.0, 10.3)
    t = ev.THRESHOLD_PRIMARY

    perfect = ev.regression_metrics(y, y)
    check("perfect predictor: RMSE 0, R2 1", abs(perfect.rmse) < 1e-12 and abs(perfect.r2 - 1) < 1e-12)

    y_tr, y_te = y[:300], y[300:]
    mean_pred = ev.mean_predictor(y_tr, len(y_te))
    mm = ev.regression_metrics(y_te, mean_pred)
    check("train-mean predictor: R2 <= 0", mm.r2 <= 1e-9, f"R2={mm.r2:.5f}")
    check("constant prediction yields NaN correlation, not a spurious value",
          np.isnan(mm.spearman) and np.isnan(mm.pearson))

    maj = ev.majority_class_predictor(y_tr, len(y_te), t)
    cm = ev.classification_metrics(y_te, maj, t)
    pos_rate = float((y_te >= t).mean())
    check("majority class: balanced accuracy 0.5, MCC 0", abs(cm.balanced_accuracy - 0.5) < 1e-12 and abs(cm.mcc) < 1e-12)
    check("majority class: PR-AUC equals base rate", abs(cm.pr_auc - pos_rate) < 1e-9,
          f"{cm.pr_auc:.4f} vs {pos_rate:.4f}")

    a = ev.bootstrap_indices(n)
    b = ev.bootstrap_indices(n)
    check("bootstrap resample matrix is shared between models", a is b and a.shape == (ev.N_BOOTSTRAP, n))

    noisy = y + rng.normal(0, 0.6, n)
    d_same = ev.paired_bootstrap_delta(y, noisy, noisy, "rmse")
    check("identical models: delta 0, p 1, not significant",
          abs(d_same.delta) < 1e-12 and abs(d_same.p_value - 1.0) < 1e-12 and not d_same.significant)

    good = y + rng.normal(0, 0.25, n)
    bad = y + rng.normal(0, 1.10, n)
    d_diff = ev.paired_bootstrap_delta(y, good, bad, "rmse", label_a="good", label_b="bad")
    check("clearly better model: significant, favours it despite lower-is-better",
          d_diff.significant and d_diff.favours == "a" and d_diff.delta < 0)

    for desc, fn in (
        ("shape mismatch", lambda: ev.regression_metrics(y, y[:10])),
        ("non-finite predictions", lambda: ev.regression_metrics(y, np.where(np.arange(n) == 3, np.nan, y))),
        ("classification metric without a threshold", lambda: ev.compute_metric("roc_auc", y, y)),
    ):
        try:
            fn()
            check(f"harness rejects {desc}", False, "no exception raised")
        except ev.EvaluationError:
            check(f"harness rejects {desc}", True)


# ---------------------------------------------------------------------------------
# 2. Split integrity, re-derived from the CSV
# ---------------------------------------------------------------------------------


def verify_split() -> pd.DataFrame:
    """Recompute the split digest and the disjointness properties from the raw artifact."""
    print("\n[2] Split integrity")
    csv = DATA / "split_assignment.csv"
    manifest = json.loads((RESULTS / "split_manifest.json").read_text(encoding="utf-8"))

    observed = sha256_file(csv)
    check("split_assignment.csv matches the manifest SHA-256",
          observed == manifest["split_assignment_sha256"], observed[:16] + "...")

    bundle = load_split(csv, RESULTS / "split_manifest.json")
    check("contract loads and reports zero test accesses on load", bundle.n_test_accesses == 0)

    split = pd.read_csv(csv)
    folds = {f: split[split.fold == f] for f in ("train", "val", "test")}
    for column in ("inchikey", "molecule_chembl_id", "canonical_smiles", "scaffold_group_id"):
        shared = 0
        names = list(folds)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared += len(set(folds[names[i]][column]) & set(folds[names[j]][column]))
        check(f"folds are disjoint on {column}", shared == 0, f"{shared} shared")

    cyclic = split[~split.is_acyclic.astype(bool)]
    cfolds = {f: cyclic[cyclic.fold == f] for f in ("train", "val", "test")}
    shared_scaffolds = len(set(cfolds["train"].murcko_scaffold) &
                           (set(cfolds["val"].murcko_scaffold) | set(cfolds["test"].murcko_scaffold)))
    check("Murcko scaffold strings are disjoint across folds (cyclic compounds)",
          shared_scaffolds == 0, f"{shared_scaffolds} shared")

    check("no duplicate canonical SMILES anywhere in the dataset",
          int(split.canonical_smiles.duplicated().sum()) == 0)
    check("fold sizes partition the dataset",
          sum(len(f) for f in folds.values()) == len(split))
    return split


# ---------------------------------------------------------------------------------
# 3. The full gate suite, re-run against the artifacts on disk
# ---------------------------------------------------------------------------------


def verify_gates(split: pd.DataFrame) -> None:
    """Re-run every data, split and feature gate rather than trusting the stored report."""
    print("\n[3] Gate suite (re-run, not re-read)")
    results = gates_mod.run_all_gates(DATA, RESULTS, write_report=False)
    results += gates_mod.run_feature_gates(split.canonical_smiles.tolist())
    for r in results:
        check(r.name, r.passed, f"observed={r.observed}",
              severity="fail" if r.severity == gates_mod.SEVERITY_FAIL else "flag")


# ---------------------------------------------------------------------------------
# 4. Protocol symmetry between the two models
# ---------------------------------------------------------------------------------


def verify_protocol() -> tuple[dict[str, dict], dict] | None:
    """Check every model shared a split, shared a harness, and was treated alike.

    Generalised over however many models are present: adding a third model must not
    quietly escape the symmetry checks that make the comparison meaningful.
    """
    print()
    print("[4] Model protocol symmetry")
    model_files = {
        "random forest": "metrics_rf.json",
        "GINE": "metrics_gnn.json",
        "Chemprop D-MPNN": "metrics_chemprop.json",
    }
    present = {n: f for n, f in model_files.items() if (RESULTS / f).exists()}
    if len(present) < 2 or not (RESULTS / "comparison.json").exists():
        print(f"  SKIP  need >=2 model artifacts plus comparison.json; found {sorted(present)}")
        return None

    models = {n: json.loads((RESULTS / f).read_text(encoding="utf-8")) for n, f in present.items()}
    comparison = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((RESULTS / "split_manifest.json").read_text(encoding="utf-8"))
    print(f"  models present: {', '.join(models)}")

    splits = {m["split_sha256"] for m in models.values()}
    check("every model consumed the identical split",
          len(splits) == 1 and splits.pop() == manifest["split_assignment_sha256"])

    harnesses = {m["evaluate_module_sha256"] for m in models.values()}
    check("every model was scored by identical harness code", len(harnesses) == 1)
    check("the harness has not changed since the models were scored",
          harnesses and next(iter({m["evaluate_module_sha256"] for m in models.values()})) == ev.module_sha256(),
          "re-run the model scripts if this fails")

    for name, m in models.items():
        check(f"{name}: 5 seeds", m["n_seeds"] == 5, f"n_seeds={m['n_seeds']}")
        check(f"{name}: refit on train + val", bool(m["refit_on_train_val"]))
        check(f"{name}: read the test fold exactly once", m["n_test_accesses"] == 1,
              f"{m['n_test_accesses']} access(es)")
    fit_sizes = {m["n_fit"] for m in models.values()}
    check("every model was fit on the same number of compounds", len(fit_sizes) == 1,
          f"{sorted(fit_sizes)}")

    check("comparison.json covers every model artifact present",
          len(comparison.get("models", [])) == len(models),
          f"comparison lists {len(comparison.get('models', []))}, artifacts present {len(models)}")
    return models, comparison


def verify_no_test_tuning() -> None:
    """Static check that no model script selects hyperparameters using the test fold."""
    print("\n[5] Static check: tuning never touches the test fold")
    scripts = [s for s in ("04_train_baseline.py", "05_train_gnn.py", "05b_train_chemprop.py")
               if (SCRIPTS / s).exists()]
    for script in scripts:
        source = (SCRIPTS / script).read_text(encoding="utf-8")
        tree = ast.parse(source)
        n_calls = sum(
            1 for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "test"
        )
        check(f"{script} calls bundle.test() exactly once", n_calls == 1, f"{n_calls} call(s)")

        # Check actual import statements via the AST, not a substring search: both
        # scripts' docstrings contain the sentence "never imports scaffold_split", and a
        # substring check flags its own documentation. The AST test is also the stricter
        # one, since it cannot be satisfied by a comment.
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
                imported.update(f"{node.module or ''}.{a.name}" for a in node.names)
        offending = sorted(n for n in imported if "scaffold_split" in n)
        check(f"{script} does not import scaffold_split", not offending,
              f"imports {offending}" if offending else "verified via AST")
        for marker in ("tune_random_forest", "TUNING_GRID", "train_once("):
            tune_line = source.find(marker)
            if tune_line != -1:
                break
        test_line = source.find("bundle.test()")
        check(f"{script} tunes before it reads the test fold",
              tune_line == -1 or test_line == -1 or tune_line < test_line)


# ---------------------------------------------------------------------------------
# 6. Every number in the README must be re-derivable from the JSON
# ---------------------------------------------------------------------------------


def verify_readme(models: dict[str, dict], comparison: dict) -> None:
    """Re-derive the README's quoted numbers from the artifacts and compare."""
    print("\n[6] README consistency")
    readme = ROOT / "README.md"
    if not readme.exists() or len(readme.read_text(encoding="utf-8")) < 500:
        print("  SKIP  README.md not written yet")
        return
    text = readme.read_text(encoding="utf-8")

    # Look the rows up by model name rather than by position: a positional index would
    # silently verify the wrong row if the table order ever changed.
    by_model = {row["model"]: row for row in comparison["table"]}
    check("comparison table contains a row for every model",
          all(label in by_model for label in comparison["models"]),
          f"models={comparison['models']}")
    recomputed = {
        f"{label} RMSE": by_model[label]["rmse"] for label in comparison["models"]
    }
    recomputed["headline delta"] = comparison["deltas"]["rmse"]["delta"]
    for key, value in recomputed.items():
        rendered = f"{value:.3f}"
        check(f"README quotes {key} = {rendered}", rendered in text or f"{value:.4f}" in text,
              f"expected {rendered} somewhere in README.md")

    verdict_recomputed = ev.verdict(ev.DeltaResult(**comparison["deltas"]["rmse"]))
    check("stored verdict matches the one recomputed from the delta CI",
          verdict_recomputed == comparison["verdict"], comparison["verdict"][:70] + "...")

    for claim, present in (
        ("censored fraction stated", "censor" in text.lower()),
        ("scaffold split described", "scaffold" in text.lower()),
        ("limitations section present", "## Limitations" in text),
        ("AI usage disclosed", (ROOT / "AI_USAGE.md").exists()),
    ):
        check(claim, present)


def main() -> None:
    print("=" * 78)
    print("VERIFICATION PASS - re-deriving every claim from the committed artifacts")
    print("=" * 78)

    verify_harness()
    split = verify_split()
    verify_gates(split)
    protocol = verify_protocol()
    verify_no_test_tuning()
    if protocol is not None:
        verify_readme(protocol[0], protocol[1])

    print("\n" + "=" * 78)
    if flags:
        print(f"{len(flags)} FLAG(S) - reported, not fatal:")
        for f in flags:
            print(f"  - {f}")
    if failures:
        print(f"\nVERIFICATION FAILED - {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("VERIFICATION PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
