"""Build the deterministic Murcko scaffold split and seal it with a SHA-256 manifest.

Reads data/hdac1_clean.csv; writes data/split_assignment.csv,
results/split_manifest.json, results/test_nn_similarity.csv and
results/gate_report.json.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rdkit  # noqa: E402
import sklearn  # noqa: E402

from src.featurize import FP_SIZE, INCLUDE_CHIRALITY, MORGAN_RADIUS  # noqa: E402
from src.gates import (  # noqa: E402
    HIGH_SIMILARITY_THRESHOLD,
    NN_SIMILARITY_NAME,
    compute_nn_similarity,
    identical_fingerprint_pairs,
    raise_on_failure,
    random_split_similarity_control,
    run_all_gates,
    similarity_summary,
)
from src.scaffold_split import (  # noqa: E402
    ALGORITHM_NAME,
    SCAFFOLD_FUNCTION_SIGNATURE,
    TRAIN_FRACTION,
    VAL_FRACTION,
    build_split,
)
from src.split_contract import (  # noqa: E402
    SHA256_MANIFEST_KEY,
    SPLIT_CSV_NAME,
    SPLIT_MANIFEST_NAME,
    load_split,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

CLEAN_CSV = DATA / "hdac1_clean.csv"
SPLIT_CSV = DATA / SPLIT_CSV_NAME
SPLIT_MANIFEST = RESULTS / SPLIT_MANIFEST_NAME
NN_SIMILARITY_CSV = RESULTS / NN_SIMILARITY_NAME

SPLIT_COLUMNS = (
    "inchikey",
    "molecule_chembl_id",
    "canonical_smiles",
    "murcko_scaffold",
    "scaffold_group_id",
    "scaffold_group_size",
    "fold",
    "pIC50",
    "n_measurements",
    "pIC50_range",
    "is_acyclic",
)


def _git_sha() -> str:
    """Return the current git commit SHA, or "uncommitted" before the first commit."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "git-unavailable"
    return completed.stdout.strip() if completed.returncode == 0 else "uncommitted"


def main() -> None:
    """Build, seal, and gate the scaffold split."""
    print("[1/5] Loading cleaned dataset ...", flush=True)
    clean = pd.read_csv(CLEAN_CSV)
    print(f"      {len(clean)} compounds", flush=True)

    print("[2/5] Building deterministic Murcko scaffold split ...", flush=True)
    split = build_split(
        clean, smiles_column="canonical_smiles",
        train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION,
    )
    for name in ("train", "val", "test"):
        print(
            f"      {name:<5} n={split.fold_sizes[name]:<6} "
            f"frac={split.fold_fractions[name]:.4f} "
            f"groups={split.fold_group_counts[name]}",
            flush=True,
        )
    print(
        f"      {split.n_groups} scaffold groups, n_acyclic={split.n_acyclic} "
        f"({split.n_acyclic / split.n_compounds:.4f})",
        flush=True,
    )

    assignment = split.frame[list(SPLIT_COLUMNS)].sort_values("inchikey").reset_index(drop=True)
    assignment.to_csv(SPLIT_CSV, index=False)
    split_sha256 = sha256_file(SPLIT_CSV)
    print(f"      wrote {SPLIT_CSV.name}; sha256={split_sha256}", flush=True)

    print("[3/5] Computing val/test -> train nearest-neighbour Tanimoto ...", flush=True)
    similarity = compute_nn_similarity(assignment)
    similarity.to_csv(NN_SIMILARITY_CSV, index=False)
    test_sims = similarity.loc[similarity["query_fold"] == "test", "max_sim_to_train"]
    print(
        f"      test median max-sim = {np.median(test_sims):.4f}; "
        f"fraction >= {HIGH_SIMILARITY_THRESHOLD} = "
        f"{float((test_sims >= HIGH_SIMILARITY_THRESHOLD).mean()):.4f}",
        flush=True,
    )

    print("[4/5] Writing split manifest ...", flush=True)
    manifest = {
        SHA256_MANIFEST_KEY: split_sha256,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "src_git_sha": _git_sha(),
        "split_csv": SPLIT_CSV.name,
        "fingerprint": {
            "radius": MORGAN_RADIUS,
            "fp_size": FP_SIZE,
            "include_chirality": INCLUDE_CHIRALITY,
            "api": "rdFingerprintGenerator.GetMorganGenerator",
        },
        "library_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "rdkit": rdkit.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        **split.to_dict(),
    }
    manifest["algorithm"] = ALGORITHM_NAME
    manifest["scaffold_function"] = SCAFFOLD_FUNCTION_SIGNATURE
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"      wrote {SPLIT_MANIFEST.name}", flush=True)

    bundle = load_split(SPLIT_CSV, manifest_json=SPLIT_MANIFEST)
    print(
        f"      contract verified: sha256 matches, fold sizes {bundle.fold_sizes}, "
        f"n_test_accesses={bundle.n_test_accesses}",
        flush=True,
    )

    print("[5/5] Running data + split gates ...", flush=True)
    control = random_split_similarity_control(assignment)
    pairs = identical_fingerprint_pairs(assignment, similarity)
    results = run_all_gates(
        DATA,
        RESULTS,
        write_report=True,
        extra={
            "created_utc": manifest["created_utc"],
            "src_git_sha": manifest["src_git_sha"],
            SHA256_MANIFEST_KEY: split_sha256,
            "library_versions": manifest["library_versions"],
            "similarity_diagnostics": {
                "scaffold_split_test_to_train": similarity_summary(test_sims.to_numpy()),
                "random_split_control_test_to_train": control,
                "note": (
                    "The random-split control uses the same fold fractions with no "
                    "chemotype separation, so it measures how congested this dataset "
                    "is. Read L10 against it, not against an absolute constant."
                ),
            },
            "identical_fingerprint_pairs": pairs.to_dict(orient="records"),
        },
    )
    for result in results:
        print(f"      {result.format_line()}", flush=True)

    print(
        f"      random-split control test->train median max-sim = {control['median']:.4f} "
        f"(scaffold split: {np.median(test_sims):.4f})",
        flush=True,
    )
    n_failed = sum(1 for r in results if not r.passed and r.severity == "fail")
    print(
        f"      artifacts written: {SPLIT_CSV.name}, {SPLIT_MANIFEST.name}, "
        f"{NN_SIMILARITY_CSV.name}, gate_report.json",
        flush=True,
    )
    if n_failed:
        print(f"      {n_failed} gate(s) failed - see gate_report.json", flush=True)
    raise_on_failure(results)
    print("03_build_split.py complete.", flush=True)


if __name__ == "__main__":
    main()
