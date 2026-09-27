"""Standardize HDAC1 structures, aggregate replicates by InChIKey, and gate the result.

Reads data/raw_activities.csv, writes data/hdac1_clean.csv and updates
data/provenance.json with the cleaning funnel.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gates import run_data_gates  # noqa: E402
from src.standardize import (  # noqa: E402
    MAX_PIC50_RANGE,
    aggregate_by_key,
    compute_pic50,
    standardize_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

RAW_ACTIVITIES_CSV = DATA / "raw_activities.csv"
CLEAN_CSV = DATA / "hdac1_clean.csv"
PROVENANCE_JSON = DATA / "provenance.json"

VALUE_COLUMN = "pIC50"
KEY_COLUMN = "inchikey"
CARRY_COLUMNS = ("molecule_chembl_id", "std_canonical_smiles", "n_heavy_atoms")

CLEAN_COLUMNS = (
    "inchikey",
    "molecule_chembl_id",
    "canonical_smiles",
    "pIC50",
    "n_measurements",
    "pIC50_range",
    "n_heavy_atoms",
)


def main() -> None:
    """Run standardization, replicate aggregation, and the data gates."""
    print("[1/5] Loading raw activities ...", flush=True)
    raw = pd.read_csv(RAW_ACTIVITIES_CSV)
    funnel: dict[str, int] = {"raw_activity_rows": len(raw)}
    print(f"      {len(raw)} rows / {raw['molecule_chembl_id'].nunique()} molecule ids", flush=True)

    print("[2/5] Standardizing structures ...", flush=True)
    standardized, reasons = standardize_frame(raw, smiles_column="canonical_smiles")
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {reason:<34} {count}", flush=True)
    funnel["standardized_ok_rows"] = len(standardized)
    funnel["unique_inchikey"] = int(standardized["inchikey"].nunique())
    n_stereocentres_stripped = int(standardized["n_stereocentres_stripped"].sum())
    n_molecules_with_stereo = int((standardized["n_stereocentres_stripped"] > 0).sum())
    print(
        f"      n_stereocentres_stripped = {n_stereocentres_stripped} "
        f"across {n_molecules_with_stereo} activity rows",
        flush=True,
    )

    print("[3/5] Computing pIC50 independently of pchembl_value ...", flush=True)
    standardized[VALUE_COLUMN] = compute_pic50(standardized["standard_value"])
    max_deviation = float(
        (standardized[VALUE_COLUMN] - standardized["pchembl_value"].astype(float)).abs().max()
    )
    print(f"      max |pIC50 - pchembl_value| = {max_deviation:.6f}", flush=True)

    print(f"[4/5] Aggregating by InChIKey (drop pIC50_range > {MAX_PIC50_RANGE}) ...", flush=True)
    aggregated, counts = aggregate_by_key(
        standardized,
        key_column=KEY_COLUMN,
        value_column=VALUE_COLUMN,
        max_range=MAX_PIC50_RANGE,
        carry_columns=CARRY_COLUMNS,
    )
    for stage, count in counts.items():
        print(f"      {stage:<34} {count}", flush=True)
    funnel.update(counts)

    clean = aggregated.rename(columns={"std_canonical_smiles": "canonical_smiles"})
    clean = clean[list(CLEAN_COLUMNS)].sort_values(KEY_COLUMN).reset_index(drop=True)
    funnel["n_final"] = len(clean)

    print("[5/5] Running data gates ...", flush=True)
    report = run_data_gates(
        clean,
        raw_with_pic50=standardized,
        max_pchembl_deviation=max_deviation,
    )
    for result in report:
        print(f"      {result.format_line()}", flush=True)

    clean.to_csv(CLEAN_CSV, index=False)
    print(f"      wrote {CLEAN_CSV.name} ({len(clean)} compounds)", flush=True)

    provenance = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8"))
    provenance["cleaning"] = {
        "standardization_pipeline": [
            "rdMolStandardize.Cleanup",
            "genuine-mixture rejection",
            "rdMolStandardize.FragmentParent",
            "rdMolStandardize.Uncharger.uncharge",
            "Chem.RemoveStereochemistry",
            "Chem.MolToSmiles / Chem.MolToInchiKey",
        ],
        "reason_counts": reasons,
        "funnel": funnel,
        "n_stereocentres_stripped": n_stereocentres_stripped,
        "n_activity_rows_with_stereo": n_molecules_with_stereo,
        "max_abs_pic50_minus_pchembl": max_deviation,
        "max_pic50_range_allowed": MAX_PIC50_RANGE,
        "pIC50_summary": {
            "min": float(clean[VALUE_COLUMN].min()),
            "median": float(clean[VALUE_COLUMN].median()),
            "max": float(clean[VALUE_COLUMN].max()),
        },
        "class_balance": {
            f"pIC50_ge_{threshold}": {
                "n_active": int((clean[VALUE_COLUMN] >= threshold).sum()),
                "n_inactive": int((clean[VALUE_COLUMN] < threshold).sum()),
                "pos_rate": round(float((clean[VALUE_COLUMN] >= threshold).mean()), 4),
            }
            for threshold in (6.0, 6.5, 7.0)
        },
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"      updated {PROVENANCE_JSON.name}", flush=True)
    print("02_clean_dataset.py complete.", flush=True)


if __name__ == "__main__":
    main()
