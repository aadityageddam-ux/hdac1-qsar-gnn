"""Deterministic RDKit standardization of SMILES into a canonical parent structure.

Pipeline: Cleanup -> FragmentParent -> Uncharger -> RemoveStereochemistry ->
canonical SMILES + InChIKey.

`FragmentParent` is used rather than `SaltRemover`: SaltRemover works from a fixed
salt list and silently leaves unlisted counterions attached, so the same parent
compound can hash to two different fingerprints. FragmentParent is list-free and
deterministically prefers the organic parent fragment.

Generic: takes SMILES strings and dataframes, not any particular target.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

# RDKit prints parse/valence warnings to stderr for every bad record; the pipeline
# records the failure reason itself, so the global stream is silenced.
RDLogger.DisableLog("rdApp.*")

# Elements plausible in a small-molecule inhibitor series. Anything outside this
# set (metals, unusual main-group atoms) is a reagent, a complex, or a data error.
ALLOWED_ELEMENTS: frozenset[str] = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
)

MIN_HEAVY_ATOMS = 5
# The 100-heavy-atom ceiling is deliberately generous: it keeps the macrocyclic
# HDAC inhibitor classes (romidepsin, trapoxin, apicidin) that a molecular-weight
# cut would discard.
MAX_HEAVY_ATOMS = 100

# If the second-largest fragment is at least this fraction of the largest, the
# record is a genuine two-component mixture rather than a salt, and choosing a
# "parent" would be arbitrary.
MIXTURE_SIZE_RATIO = 0.8

# pIC50 = 9 - log10(IC50 in nM); i.e. -log10(IC50 in M).
PIC50_OFFSET = 9.0

# Published inter-laboratory reproducibility of ChEMBL IC50 data is ~0.5-0.7 log
# SD (Kramer 2012, Kalliokoski 2013). 1.0 log is about 2 SD: beyond that, the
# replicate measurements are not measurements of the same quantity.
MAX_PIC50_RANGE = 1.0

_UNCHARGER = rdMolStandardize.Uncharger()


class StandardizationError(RuntimeError):
    """Raised when standardization is asked to do something structurally impossible."""


@dataclass(frozen=True)
class StandardizedMolecule:
    """Result of running one input SMILES through the standardization pipeline."""

    input_smiles: str
    ok: bool
    reason: str
    canonical_smiles: str | None
    inchikey: str | None
    n_heavy_atoms: int
    n_stereocentres_stripped: int

    def to_dict(self) -> dict:
        """Return the standardization result as a plain JSON-serialisable dict."""
        return asdict(self)


def _failure(input_smiles: str, reason: str) -> StandardizedMolecule:
    """Build a rejected StandardizedMolecule with the given reason code."""
    return StandardizedMolecule(
        input_smiles=input_smiles,
        ok=False,
        reason=reason,
        canonical_smiles=None,
        inchikey=None,
        n_heavy_atoms=0,
        n_stereocentres_stripped=0,
    )


def _count_stereo(mol: Chem.Mol) -> int:
    """Count assigned atom chiral tags plus directional double bonds on a molecule."""
    n_atoms = sum(
        1 for atom in mol.GetAtoms() if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    n_bonds = sum(
        1 for bond in mol.GetBonds() if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return n_atoms + n_bonds


def _is_genuine_mixture(mol: Chem.Mol) -> bool:
    """True when the two largest fragments are within MIXTURE_SIZE_RATIO of each other."""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(fragments) < 2:
        return False
    sizes = sorted((frag.GetNumHeavyAtoms() for frag in fragments), reverse=True)
    if sizes[0] == 0:
        return False
    return sizes[1] / sizes[0] >= MIXTURE_SIZE_RATIO


def standardize_smiles(smiles: str) -> StandardizedMolecule:
    """Standardize one SMILES to a canonical, desalted, uncharged, stereo-free parent."""
    if not isinstance(smiles, str) or not smiles.strip():
        return _failure(str(smiles), "empty_smiles")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _failure(smiles, "parse_failure")

    try:
        mol = rdMolStandardize.Cleanup(mol)
    except Exception:  # RDKit raises bare RuntimeError/ValueError on pathological input
        return _failure(smiles, "cleanup_failure")
    if mol is None:
        return _failure(smiles, "cleanup_failure")

    # Checked before FragmentParent: after it, only one fragment is left and the
    # evidence that the record was a two-component mixture has been discarded.
    if _is_genuine_mixture(mol):
        return _failure(smiles, "genuine_mixture")

    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return _failure(smiles, "fragment_or_uncharge_failure")
    if mol is None or mol.GetNumAtoms() == 0:
        return _failure(smiles, "empty_after_fragment_parent")

    n_stereo = _count_stereo(mol)
    Chem.RemoveStereochemistry(mol)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return _failure(smiles, "sanitize_failure")

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < MIN_HEAVY_ATOMS:
        return _failure(smiles, "too_few_heavy_atoms")
    if n_heavy > MAX_HEAVY_ATOMS:
        return _failure(smiles, "too_many_heavy_atoms")

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not elements <= ALLOWED_ELEMENTS:
        return _failure(smiles, "disallowed_element")

    canonical = Chem.MolToSmiles(mol)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        return _failure(smiles, "inchikey_failure")

    return StandardizedMolecule(
        input_smiles=smiles,
        ok=True,
        reason="ok",
        canonical_smiles=canonical,
        inchikey=inchikey,
        n_heavy_atoms=n_heavy,
        n_stereocentres_stripped=n_stereo,
    )


def standardize_frame(
    frame: pd.DataFrame, smiles_column: str = "canonical_smiles"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardize every row of `frame`, returning the surviving rows and a reason tally."""
    if smiles_column not in frame.columns:
        raise StandardizationError(f"Column {smiles_column!r} not in {list(frame.columns)}")

    # The same input SMILES recurs across replicate measurements; standardizing
    # each distinct string once turns ~9k calls into ~7k.
    cache: dict[str, StandardizedMolecule] = {}
    records: list[StandardizedMolecule] = []
    for smiles in frame[smiles_column].tolist():
        key = smiles if isinstance(smiles, str) else ""
        if key not in cache:
            cache[key] = standardize_smiles(key)
        records.append(cache[key])

    reasons: dict[str, int] = {}
    for record in records:
        reasons[record.reason] = reasons.get(record.reason, 0) + 1

    out = frame.copy()
    out["std_canonical_smiles"] = [r.canonical_smiles for r in records]
    out["inchikey"] = [r.inchikey for r in records]
    out["n_heavy_atoms"] = [r.n_heavy_atoms for r in records]
    out["n_stereocentres_stripped"] = [r.n_stereocentres_stripped for r in records]
    out["std_ok"] = [r.ok for r in records]
    out["std_reason"] = [r.reason for r in records]

    kept = out[out["std_ok"]].drop(columns=["std_ok"]).reset_index(drop=True)
    return kept, reasons


def compute_pic50(value_nm: pd.Series) -> pd.Series:
    """Compute pIC50 = 9 - log10(IC50 in nM) independently of ChEMBL's pchembl_value."""
    if (value_nm <= 0).any():
        raise StandardizationError("compute_pic50 received a non-positive IC50 value.")
    return PIC50_OFFSET - value_nm.astype(float).map(math.log10)


def aggregate_by_key(
    frame: pd.DataFrame,
    key_column: str,
    value_column: str,
    max_range: float = MAX_PIC50_RANGE,
    carry_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse replicate measurements to a per-key median, dropping inconsistent groups.

    The median, not the mean, is used: it is robust to a single transcription error
    in a replicate set, which the mean is not.
    """
    for column in (key_column, value_column, *carry_columns):
        if column not in frame.columns:
            raise StandardizationError(f"Column {column!r} not in {list(frame.columns)}")

    grouped = frame.groupby(key_column, sort=True)
    aggregated = pd.DataFrame(
        {
            value_column: grouped[value_column].median(),
            "n_measurements": grouped[value_column].size(),
            f"{value_column}_range": grouped[value_column].max()
            - grouped[value_column].min(),
        }
    )
    for column in carry_columns:
        # First row in a deterministic sort order: replicates of one InChIKey may
        # carry different ChEMBL ids or input SMILES, and one must be chosen.
        aggregated[column] = grouped[column].min()

    aggregated = aggregated.reset_index()
    n_groups = len(aggregated)

    range_column = f"{value_column}_range"
    consistent = aggregated[aggregated[range_column] <= max_range].reset_index(drop=True)

    counts = {
        "n_rows_in": len(frame),
        "n_groups": n_groups,
        "n_groups_dropped_range": n_groups - len(consistent),
        "n_groups_kept": len(consistent),
        "n_groups_with_replicates": int((aggregated["n_measurements"] > 1).sum()),
    }
    return consistent, counts


def count_rings(smiles: str) -> int:
    """Return the number of rings in a SMILES string, for acyclic-fraction diagnostics."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StandardizationError(f"Could not parse standardized SMILES {smiles!r}")
    return rdMolDescriptors.CalcNumRings(mol)
