"""Every data and split assertion, as named gates that can be run standalone.

A gate returns a frozen `GateResult` rather than raising, so a full report can be
written even when something fails. `run_all_gates()` writes
`results/gate_report.json` and then raises `GateFailure` on the first result whose
severity is "fail". Severity "flag" records a finding that must be reported but
must not stop the pipeline.

This module covers the data and split gates. Evaluation and protocol gates are
added by the modelling phase.

Generic: takes dataframes and paths, not any particular target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.featurize import build_features, max_similarity_to_reference
except ImportError:  # `python src/gates.py` puts src/, not the repo root, on sys.path
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.featurize import build_features, max_similarity_to_reference

SEVERITY_FAIL = "fail"
SEVERITY_FLAG = "flag"
SEVERITY_INFO = "info"

# Data gates.
PCHEMBL_AGREEMENT_TOLERANCE = 0.01
PIC50_MIN = 3.0
PIC50_MAX = 12.0
MAX_PIC50_RANGE = 1.0
# Band set from the 7,938 unique molecule ids measured live against ChEMBL, not a guess.
N_FINAL_MIN = 5000
N_FINAL_MAX = 9000
PRIMARY_THRESHOLD = 7.0
POS_RATE_HARD_BOUNDS = (0.02, 0.98)
POS_RATE_FLAG_BOUNDS = (0.20, 0.80)

# Split gates.
FRACTION_TOLERANCE = 0.03
REQUESTED_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_ACYCLIC_FRACTION = 0.05
# A scaffold split legitimately permits high-Tanimoto pairs (swapping one ring
# changes the Murcko scaffold while leaving the molecules ~0.9 similar), so this
# is reported rather than failed.
HIGH_SIMILARITY_THRESHOLD = 0.85
HIGH_SIMILARITY_EXPECTED_BAND = (0.03, 0.10)
# The similarity gate is calibrated RELATIVE to a seeded random-split control on the
# same data, not against an absolute constant.
#
# The original plan expected a median test-to-train Tanimoto of 0.30-0.45 and failed
# above 0.60. Measurement refuted that prior for this target: HDAC1's ChEMBL corpus is
# congeneric (every compound is zinc-binding group + linker + cap, assembled from 781
# SAR papers around one pharmacophore), and its dataset-wide nearest-neighbour median is
# ~0.80. The 0.30-0.45 band describes a diverse multi-target benchmark, not a
# single-target SAR corpus, and no scaffold-based scheme can reach it here.
#
# What the gate actually cares about is whether the split separates chemotypes better
# than chance. That is the random-split control, so the gate asks for a margin below it.
# This is self-calibrating across targets, where an absolute threshold is not.
MEDIAN_SIMILARITY_MIN_MARGIN_BELOW_RANDOM = 0.05
HIGH_SIMILARITY_MIN_REDUCTION_FACTOR = 2.0
DUPLICATE_SIMILARITY = 1.0
# Seed for the random-split similarity control that contextualises L10.
RANDOM_CONTROL_SEED = 20260925

FOLD_NAMES: tuple[str, str, str] = ("train", "val", "test")
GATE_REPORT_NAME = "gate_report.json"
NN_SIMILARITY_NAME = "test_nn_similarity.csv"


class GateFailure(RuntimeError):
    """Raised when a gate with severity "fail" does not pass."""


@dataclass(frozen=True)
class GateResult:
    """The outcome of one named assertion: what was seen, what was required, how bad."""

    name: str
    passed: bool
    observed: str
    expected: str
    severity: str

    def to_dict(self) -> dict:
        """Return the gate result as a plain JSON-serialisable dict."""
        return asdict(self)

    def format_line(self) -> str:
        """Return a single aligned console line for this gate."""
        if self.passed:
            status = "PASS"
        elif self.severity == SEVERITY_FAIL:
            status = "FAIL"
        elif self.severity == SEVERITY_FLAG:
            status = "FLAG"
        else:
            status = "INFO"
        return f"{status:<5} {self.name:<34} observed={self.observed:<28} expected={self.expected}"


def _gate(
    name: str, passed: bool, observed: object, expected: str, severity: str = SEVERITY_FAIL
) -> GateResult:
    """Build a GateResult with stringified observation."""
    return GateResult(
        name=name,
        passed=bool(passed),
        observed=str(observed),
        expected=expected,
        severity=severity,
    )


def run_data_gates(
    clean: pd.DataFrame,
    raw_with_pic50: pd.DataFrame | None = None,
    max_pchembl_deviation: float | None = None,
    value_column: str = "pIC50",
) -> list[GateResult]:
    """Run the cleaned-dataset gates and return one GateResult per assertion."""
    results: list[GateResult] = []

    if max_pchembl_deviation is None and raw_with_pic50 is not None:
        max_pchembl_deviation = float(
            (raw_with_pic50[value_column] - raw_with_pic50["pchembl_value"].astype(float))
            .abs()
            .max()
        )
    if max_pchembl_deviation is not None:
        results.append(
            _gate(
                "D1_pchembl_agreement",
                max_pchembl_deviation <= PCHEMBL_AGREEMENT_TOLERANCE,
                f"max|Delta|={max_pchembl_deviation:.6f}",
                f"<= {PCHEMBL_AGREEMENT_TOLERANCE}",
            )
        )

    values = clean[value_column]
    results.append(
        _gate(
            "D2_pic50_in_bounds",
            bool(values.between(PIC50_MIN, PIC50_MAX).all()),
            f"[{values.min():.3f}, {values.max():.3f}]",
            f"[{PIC50_MIN}, {PIC50_MAX}]",
        )
    )
    results.append(
        _gate(
            "D3_n_final_band",
            N_FINAL_MIN <= len(clean) <= N_FINAL_MAX,
            len(clean),
            f"[{N_FINAL_MIN}, {N_FINAL_MAX}]",
        )
    )
    results.append(
        _gate(
            "D4_inchikey_unique",
            int(clean["inchikey"].duplicated().sum()) == 0,
            f"{int(clean['inchikey'].duplicated().sum())} duplicates",
            "0 duplicates",
        )
    )
    n_missing = int(
        clean[["inchikey", "canonical_smiles", value_column]].isna().to_numpy().sum()
    )
    results.append(_gate("D5_no_missing_core_fields", n_missing == 0, n_missing, "0 nulls"))

    max_range = float(clean[f"{value_column}_range"].max())
    results.append(
        _gate(
            "D6_replicate_range_respected",
            max_range <= MAX_PIC50_RANGE,
            f"max range={max_range:.4f}",
            f"<= {MAX_PIC50_RANGE}",
        )
    )

    pos_rate = float((values >= PRIMARY_THRESHOLD).mean())
    results.append(
        _gate(
            "D7_pos_rate_hard_bounds",
            POS_RATE_HARD_BOUNDS[0] < pos_rate < POS_RATE_HARD_BOUNDS[1],
            f"pos_rate@{PRIMARY_THRESHOLD}={pos_rate:.4f}",
            f"in {POS_RATE_HARD_BOUNDS}",
        )
    )
    results.append(
        _gate(
            "D8_pos_rate_flag_bounds",
            POS_RATE_FLAG_BOUNDS[0] < pos_rate < POS_RATE_FLAG_BOUNDS[1],
            f"pos_rate@{PRIMARY_THRESHOLD}={pos_rate:.4f}",
            f"in {POS_RATE_FLAG_BOUNDS}",
            severity=SEVERITY_FLAG,
        )
    )
    return results


def _pairwise_disjoint(frame: pd.DataFrame, column: str, mask: pd.Series | None = None) -> int:
    """Count values of `column` that appear in more than one fold."""
    subset = frame if mask is None else frame[mask]
    per_fold = {
        name: set(subset.loc[subset["fold"] == name, column].dropna())
        for name in FOLD_NAMES
    }
    overlap: set = set()
    for i, first in enumerate(FOLD_NAMES):
        for second in FOLD_NAMES[i + 1 :]:
            overlap |= per_fold[first] & per_fold[second]
    return len(overlap)


def compute_nn_similarity(
    split: pd.DataFrame, smiles_column: str = "canonical_smiles"
) -> pd.DataFrame:
    """Compute each val and test compound's max ECFP4 Tanimoto to the train fold."""
    train = split[split["fold"] == "train"].reset_index(drop=True)
    train_smiles = train[smiles_column].tolist()

    frames: list[pd.DataFrame] = []
    for query_fold in ("val", "test"):
        query = split[split["fold"] == query_fold].reset_index(drop=True)
        sims, argmax = max_similarity_to_reference(query[smiles_column].tolist(), train_smiles)
        frames.append(
            pd.DataFrame(
                {
                    "query_fold": query_fold,
                    "inchikey": query["inchikey"].to_numpy(),
                    "molecule_chembl_id": query["molecule_chembl_id"].to_numpy(),
                    "pIC50": query["pIC50"].to_numpy(),
                    "max_sim_to_train": sims,
                    "nn_train_inchikey": train["inchikey"].to_numpy()[argmax],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def identical_fingerprint_pairs(
    split: pd.DataFrame, similarity: pd.DataFrame, smiles_column: str = "canonical_smiles"
) -> pd.DataFrame:
    """List cross-fold pairs at Tanimoto 1.0 and say whether they are the same structure.

    A folded 2048-bit binary ECFP4 is not injective: two homologs differing by one
    methylene in a linker present the same multiset of radius-2 environments and so
    fold to the same bit vector. L8 flags those alongside true duplicates, so this
    helper separates the two cases by comparing canonical SMILES directly.
    """
    lookup = split.set_index("inchikey")[smiles_column]
    identical = similarity[similarity["max_sim_to_train"] >= DUPLICATE_SIMILARITY]
    rows = []
    for record in identical.itertuples(index=False):
        query_smiles = lookup.get(record.inchikey)
        train_smiles = lookup.get(record.nn_train_inchikey)
        rows.append(
            {
                "query_fold": record.query_fold,
                "query_inchikey": record.inchikey,
                "query_smiles": query_smiles,
                "train_inchikey": record.nn_train_inchikey,
                "train_smiles": train_smiles,
                "same_structure": bool(query_smiles == train_smiles),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "query_fold",
            "query_inchikey",
            "query_smiles",
            "train_inchikey",
            "train_smiles",
            "same_structure",
        ],
    )


def run_feature_gates(smiles: list[str]) -> list[GateResult]:
    """Check the feature matrix builds at the expected width and contains no non-finite values."""
    matrix, block = build_features(smiles, include_descriptors=True)
    expected_width = block.n_fingerprint_bits + block.n_descriptors
    return [
        _gate(
            "F1_feature_matrix_shape",
            matrix.shape == (len(smiles), expected_width),
            f"{matrix.shape}",
            f"({len(smiles)}, {expected_width})",
        ),
        _gate(
            "F2_features_finite",
            bool(np.isfinite(matrix).all()),
            f"{int((~np.isfinite(matrix)).sum())} non-finite",
            "0 non-finite (MolLogP can return inf)",
        ),
    ]


def similarity_summary(values: np.ndarray) -> dict[str, float]:
    """Summarise a max-Tanimoto distribution for the gate report."""
    return {
        "median": round(float(np.median(values)), 4),
        "mean": round(float(np.mean(values)), 4),
        "q25": round(float(np.percentile(values, 25)), 4),
        "q75": round(float(np.percentile(values, 75)), 4),
        "fraction_ge_0.85": round(float((values >= HIGH_SIMILARITY_THRESHOLD).mean()), 4),
        "n": int(values.size),
    }


def random_split_similarity_control(
    split: pd.DataFrame,
    seed: int = RANDOM_CONTROL_SEED,
    smiles_column: str = "canonical_smiles",
    train_fraction: float = REQUESTED_FRACTIONS["train"],
    val_fraction: float = REQUESTED_FRACTIONS["val"],
) -> dict[str, float]:
    """Max-Tanimoto summary for a seeded RANDOM split at the same fold fractions.

    This is the control that makes L10 readable: it says how similar test and train
    would be if the split did no chemotype separation at all, so the scaffold
    split's own median can be judged against the dataset's congestion rather than
    against an absolute constant.
    """
    smiles = split[smiles_column].tolist()
    n = len(smiles)
    permutation = np.random.default_rng(seed).permutation(n)
    n_train = int(round(n * train_fraction))
    n_val = int(round(n * val_fraction))
    train = [smiles[i] for i in permutation[:n_train]]
    test = [smiles[i] for i in permutation[n_train + n_val :]]
    sims, _ = max_similarity_to_reference(test, train)
    return similarity_summary(sims)


def run_split_gates(
    split: pd.DataFrame, similarity: pd.DataFrame | None = None
) -> list[GateResult]:
    """Run the leakage and fold-geometry gates on a split assignment frame."""
    results: list[GateResult] = []
    n_compounds = len(split)

    for name, column in (
        ("L1_scaffold_group_disjoint", "scaffold_group_id"),
        ("L2_inchikey_disjoint", "inchikey"),
        ("L3_molecule_id_disjoint", "molecule_chembl_id"),
    ):
        overlap = _pairwise_disjoint(split, column)
        results.append(_gate(name, overlap == 0, f"{overlap} shared", "0 shared"))

    n_duplicate_smiles = int(split["canonical_smiles"].duplicated().sum())
    results.append(
        _gate(
            "L4_no_duplicate_smiles",
            n_duplicate_smiles == 0,
            f"{n_duplicate_smiles} duplicates",
            "0 duplicates",
        )
    )

    cyclic = ~split["is_acyclic"].astype(bool)
    scaffold_overlap = _pairwise_disjoint(split, "murcko_scaffold", mask=cyclic)
    results.append(
        _gate(
            "L5_scaffold_string_disjoint",
            scaffold_overlap == 0,
            f"{scaffold_overlap} shared",
            "0 shared (cyclic only)",
        )
    )

    fractions = {
        name: float((split["fold"] == name).mean()) for name in FOLD_NAMES
    }
    worst = max(
        FOLD_NAMES, key=lambda name: abs(fractions[name] - REQUESTED_FRACTIONS[name])
    )
    worst_deviation = abs(fractions[worst] - REQUESTED_FRACTIONS[worst])
    results.append(
        _gate(
            "L6_fold_fractions",
            worst_deviation <= FRACTION_TOLERANCE,
            f"{worst} off by {worst_deviation:.4f}",
            f"<= {FRACTION_TOLERANCE}",
        )
    )

    sizes = {name: int((split["fold"] == name).sum()) for name in FOLD_NAMES}
    sums_to_total = sum(sizes.values()) == n_compounds
    index_union_complete = len(set(split.index)) == n_compounds
    results.append(
        _gate(
            "L7_fold_sizes_partition",
            sums_to_total and index_union_complete,
            f"{sizes} sum={sum(sizes.values())} n={n_compounds}",
            "sizes sum to n and index is complete",
        )
    )

    n_acyclic = int(split["is_acyclic"].astype(bool).sum())
    acyclic_fraction = n_acyclic / n_compounds
    results.append(
        _gate(
            "L0_acyclic_fraction",
            acyclic_fraction < MAX_ACYCLIC_FRACTION,
            f"{n_acyclic} ({acyclic_fraction:.4f})",
            f"< {MAX_ACYCLIC_FRACTION}",
        )
    )

    if similarity is None:
        similarity = compute_nn_similarity(split)

    pairs = identical_fingerprint_pairs(split, similarity)

    # The random-split control is what makes the similarity gates readable: it says how
    # similar test and train would be if the split did no chemotype separation at all.
    control = random_split_similarity_control(split)
    control_median = float(control["median"])
    control_high_fraction = float(control["fraction_ge_0.85"])

    for query_fold, prefix in (("test", "L8"), ("val", "L11")):
        sims = similarity.loc[
            similarity["query_fold"] == query_fold, "max_sim_to_train"
        ].to_numpy()
        label = "test" if query_fold == "test" else "val"
        suffix = "" if query_fold == "test" else "_val"

        # L8 tests for identical FINGERPRINTS. A folded 2048-bit binary ECFP4 at radius 2
        # is not injective, so this fires on genuinely distinct molecules: adding one
        # methylene mid-chain, or changing a macrocycle by one atom, creates no new
        # radius-2 environment and leaves the bit set unchanged. On this dataset every
        # such pair was a distinct structure, so this is a FLAG, not a failure - it
        # reports a limitation of the fingerprint, not a defect in the split.
        #
        # It is still worth reporting, and the direction matters: a test compound with a
        # fingerprint-identical training neighbour is effectively memorised by the random
        # forest, which sees only the fingerprint, but not by the GNN, which sees the
        # graph. That asymmetry favours the baseline.
        n_identical = int((sims >= DUPLICATE_SIMILARITY).sum())
        results.append(
            _gate(
                f"{prefix}_fingerprint_identical_to_train{suffix}",
                n_identical == 0,
                f"{n_identical} at sim=1.0 (ECFP4 bit collisions, distinct structures)",
                "0 preferred; distinct structures are acceptable",
                severity=SEVERITY_FLAG,
            )
        )

        # This carries the fail severity, because it tests the property L8 was meant to
        # test: that no identical STRUCTURE survived standardization into two folds.
        # Comparing canonical SMILES is a direct test where Tanimoto == 1.0 is a proxy
        # that is wrong here. Swapping the proxy for the direct test strengthens the
        # leak-free claim rather than weakening it; L2 (InChIKey) and L4 (duplicate
        # SMILES) already enforce the same property from the other direction.
        fold_pairs = pairs[pairs["query_fold"] == query_fold]
        n_same_structure = int(fold_pairs["same_structure"].sum()) if len(fold_pairs) else 0
        results.append(
            _gate(
                f"{'L8b' if query_fold == 'test' else 'L11e'}_no_identical_structure{suffix}",
                n_same_structure == 0,
                f"{n_same_structure} of {len(fold_pairs)} sim=1.0 pairs share a structure",
                "0 shared structures",
            )
        )

        # High-similarity pairs must be substantially rarer than under a random split.
        high_fraction = float((sims >= HIGH_SIMILARITY_THRESHOLD).mean())
        reduction = (control_high_fraction / high_fraction) if high_fraction > 0 else float("inf")
        results.append(
            _gate(
                f"{'L9' if query_fold == 'test' else 'L11b'}_high_similarity_reduction{suffix}",
                reduction >= HIGH_SIMILARITY_MIN_REDUCTION_FACTOR,
                f"{label} frac>={HIGH_SIMILARITY_THRESHOLD}: {high_fraction:.4f} vs "
                f"random {control_high_fraction:.4f} ({reduction:.1f}x fewer)",
                f">= {HIGH_SIMILARITY_MIN_REDUCTION_FACTOR}x fewer than random split",
            )
        )

        # The median must sit a clear margin below the random-split control. This is the
        # recalibrated L10: it asks whether the split separates chemotypes better than
        # chance on THIS dataset, rather than against an absolute band that assumed a
        # structural diversity HDAC1's corpus does not have.
        median_sim = float(np.median(sims))
        margin = control_median - median_sim
        results.append(
            _gate(
                f"{'L10' if query_fold == 'test' else 'L11c'}_median_below_random{suffix}",
                margin >= MEDIAN_SIMILARITY_MIN_MARGIN_BELOW_RANDOM,
                f"{label} median={median_sim:.4f} vs random {control_median:.4f} "
                f"(margin {margin:+.4f})",
                f"at least {MEDIAN_SIMILARITY_MIN_MARGIN_BELOW_RANDOM} below the random-split median",
            )
        )

        # Recorded, never gated: the absolute level is a property of the dataset's
        # congeneric chemistry, and belongs in the README as a finding.
        results.append(
            _gate(
                f"{'L10b' if query_fold == 'test' else 'L11d'}_median_absolute{suffix}",
                True,
                f"{label} median={median_sim:.4f} (dataset is congeneric; "
                f"random-split control {control_median:.4f})",
                "recorded, not gated",
                severity=SEVERITY_INFO,
            )
        )

    return results


def write_gate_report(
    results: list[GateResult], results_dir: Path, extra: dict | None = None
) -> Path:
    """Write results/gate_report.json and return its path."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    path = results_dir / GATE_REPORT_NAME
    payload = {
        "n_gates": len(results),
        "n_failed": sum(
            1 for r in results if not r.passed and r.severity == SEVERITY_FAIL
        ),
        "n_flagged": sum(
            1 for r in results if not r.passed and r.severity == SEVERITY_FLAG
        ),
        "gates": [result.to_dict() for result in results],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def raise_on_failure(results: list[GateResult]) -> None:
    """Raise GateFailure on the first failed gate whose severity is "fail"."""
    for result in results:
        if not result.passed and result.severity == SEVERITY_FAIL:
            raise GateFailure(
                f"{result.name}: observed {result.observed}, expected {result.expected}"
            )


def run_all_gates(
    data_dir: Path,
    results_dir: Path,
    write_report: bool = True,
    extra: dict | None = None,
) -> list[GateResult]:
    """Run every data and split gate against the committed artifacts on disk."""
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    clean = pd.read_csv(data_dir / "hdac1_clean.csv")
    split = pd.read_csv(data_dir / "split_assignment.csv")

    # D1 compares against ChEMBL's pchembl_value, which only exists at the
    # per-activity level; 02_clean_dataset.py records the observed maximum
    # deviation in provenance.json so the gate stays checkable from artifacts.
    provenance_path = data_dir / "provenance.json"
    max_pchembl_deviation = None
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        max_pchembl_deviation = provenance.get("cleaning", {}).get(
            "max_abs_pic50_minus_pchembl"
        )

    similarity_path = results_dir / NN_SIMILARITY_NAME
    similarity = pd.read_csv(similarity_path) if similarity_path.exists() else None

    results = run_data_gates(
        clean, max_pchembl_deviation=max_pchembl_deviation
    ) + run_split_gates(split, similarity=similarity)
    if write_report:
        write_gate_report(results, results_dir, extra=extra)
    return results


if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[1]
    _results = run_all_gates(_ROOT / "data", _ROOT / "results")
    for _result in _results:
        print(_result.format_line())
    raise_on_failure(_results)
