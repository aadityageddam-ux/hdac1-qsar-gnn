"""Fetch HDAC1 (CHEMBL325) IC50 bioactivity from ChEMBL, plus the censored records.

Writes data/raw_activities.csv, data/hdac1_censored.csv and data/provenance.json.
All HDAC1-specific parameters live here; src/chembl_fetch.py stays target-agnostic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chembl_fetch import (  # noqa: E402
    ChemblFetchError,
    fetch_activities,
    fetch_assays,
    fetch_target,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

TARGET_CHEMBL_ID = "CHEMBL325"
EXPECTED_TARGET_NAME = "Histone deacetylase 1"
EXPECTED_ORGANISM = "Homo sapiens"
EXPECTED_UNIPROT = "Q13547"

STANDARD_TYPE = "IC50"
ASSAY_TYPE = "B"
STANDARD_UNITS = "nM"
KEPT_RELATION = "="
CENSORED_RELATIONS = (">", ">=", "<", "<=")

# Assay confidence 9 = direct single-protein target assignment. For this target
# only {9, 8} exist, so ">= 8" is a no-op and "== 9" is the meaningful filter:
# it drops the homology-assigned tail.
MIN_CONFIDENCE_SCORE = 9

RAW_ACTIVITIES_CSV = DATA / "raw_activities.csv"
CENSORED_CSV = DATA / "hdac1_censored.csv"
PROVENANCE_JSON = DATA / "provenance.json"


def _uniprot_accessions(target: dict) -> list[str]:
    """Pull UniProt accessions out of a ChEMBL target record's component list."""
    accessions = []
    for component in target.get("target_components", []):
        accession = component.get("accession")
        if accession:
            accessions.append(accession)
    return accessions


def main() -> None:
    """Run the full ChEMBL retrieval and write the raw artifacts."""
    print(f"[1/5] Confirming target {TARGET_CHEMBL_ID} ...", flush=True)
    target = fetch_target(TARGET_CHEMBL_ID)
    accessions = _uniprot_accessions(target)
    print(
        f"      {target['pref_name']} | {target['organism']} | "
        f"{target['target_type']} | UniProt {accessions}",
        flush=True,
    )
    if target["pref_name"] != EXPECTED_TARGET_NAME or target["organism"] != EXPECTED_ORGANISM:
        raise ChemblFetchError(
            f"{TARGET_CHEMBL_ID} is {target['pref_name']} / {target['organism']}, "
            f"expected {EXPECTED_TARGET_NAME} / {EXPECTED_ORGANISM}"
        )
    if EXPECTED_UNIPROT not in accessions:
        raise ChemblFetchError(f"UniProt {EXPECTED_UNIPROT} absent from {accessions}")

    print("[2/5] Fetching exact-relation IC50 activities ...", flush=True)
    activities, activity_prov = fetch_activities(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type=STANDARD_TYPE,
        assay_type=ASSAY_TYPE,
        standard_units=STANDARD_UNITS,
        standard_relation=KEPT_RELATION,
        require_pchembl=True,
    )
    n_fetched = len(activities)
    n_molecules_fetched = activities["molecule_chembl_id"].nunique()
    print(
        f"      {n_fetched} activities / {n_molecules_fetched} molecules / "
        f"{activities['assay_chembl_id'].nunique()} assays / "
        f"{activities['document_chembl_id'].nunique()} documents",
        flush=True,
    )

    print("[3/5] Fetching assay confidence scores ...", flush=True)
    assays, assay_prov = fetch_assays(activities["assay_chembl_id"].dropna().unique().tolist())
    confidence_counts = (
        assays["confidence_score"].value_counts().sort_index().to_dict()
    )
    print(f"      confidence score distribution: {confidence_counts}", flush=True)

    print("[4/5] Applying organism / value / confidence filters ...", flush=True)
    merged = activities.merge(
        assays[["assay_chembl_id", "confidence_score"]], on="assay_chembl_id", how="left"
    )
    merged["standard_value"] = merged["standard_value"].astype(float)

    funnel: dict[str, int] = {"fetched_exact_relation_pchembl": len(merged)}

    step = merged[merged["target_organism"] == EXPECTED_ORGANISM]
    funnel["organism_homo_sapiens"] = len(step)

    step = step[step["standard_value"] > 0]
    funnel["standard_value_gt_0"] = len(step)

    step = step[step["confidence_score"] == MIN_CONFIDENCE_SCORE]
    funnel["confidence_score_eq_9"] = len(step)

    step = step[step["canonical_smiles"].notna() & (step["canonical_smiles"] != "")]
    funnel["has_smiles"] = len(step)

    kept = step.reset_index(drop=True)
    funnel["unique_molecule_chembl_id"] = int(kept["molecule_chembl_id"].nunique())

    for stage, count in funnel.items():
        print(f"      {stage:<34} {count}", flush=True)

    kept.to_csv(RAW_ACTIVITIES_CSV, index=False)
    print(f"      wrote {RAW_ACTIVITIES_CSV.name} ({len(kept)} rows)", flush=True)

    print("[5/5] Fetching censored (non-exact relation) activities ...", flush=True)
    censored, censored_prov = fetch_activities(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type=STANDARD_TYPE,
        assay_type=ASSAY_TYPE,
        standard_units=STANDARD_UNITS,
        standard_relation_in=CENSORED_RELATIONS,
        require_pchembl=False,
    )
    censored.to_csv(CENSORED_CSV, index=False)
    n_censored = len(censored)
    censored_fraction = n_censored / (n_censored + len(kept))
    print(
        f"      {n_censored} censored records; "
        f"n_censored / (n_censored + n_kept) = {censored_fraction:.4f}",
        flush=True,
    )

    provenance = {
        "target": {
            "target_chembl_id": TARGET_CHEMBL_ID,
            "pref_name": target["pref_name"],
            "organism": target["organism"],
            "target_type": target["target_type"],
            "uniprot_accessions": accessions,
        },
        "filters": {
            "standard_type": STANDARD_TYPE,
            "assay_type": ASSAY_TYPE,
            "standard_units": STANDARD_UNITS,
            "standard_relation": KEPT_RELATION,
            "pchembl_value__isnull": "false",
            "target_organism": EXPECTED_ORGANISM,
            "standard_value": "> 0",
            "confidence_score": f"== {MIN_CONFIDENCE_SCORE}",
        },
        "funnel": funnel,
        "assay_confidence_score_counts": {str(k): int(v) for k, v in confidence_counts.items()},
        "n_assays": int(kept["assay_chembl_id"].nunique()),
        "n_documents": int(kept["document_chembl_id"].nunique()),
        "n_data_validity_flagged": int(kept["data_validity_comment"].notna().sum()),
        "censored": {
            "relations": list(CENSORED_RELATIONS),
            "n_censored": n_censored,
            "n_kept": len(kept),
            "censored_fraction": round(censored_fraction, 6),
        },
        "fetch_provenance": [
            activity_prov.to_dict(),
            assay_prov.to_dict(),
            censored_prov.to_dict(),
        ],
    }
    PROVENANCE_JSON.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"      wrote {PROVENANCE_JSON.name}", flush=True)
    print("01_fetch_chembl.py complete.", flush=True)


if __name__ == "__main__":
    main()
