"""ECFP4 fingerprints, 12 descriptors, and Tanimoto helpers.

Uses the current rdFingerprintGenerator API rather than the deprecated
GetMorganFingerprintAsBitVect. Binary fingerprints, not counts, so the model features and
the leakage Tanimoto check are computed on the same representation.

No scaling. A forest doesn't care about monotone per-feature transforms, so a
StandardScaler would change nothing about the model and would only add a fitted object
that could leak statistics across folds.

Generic - takes SMILES lists, not a target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdMolDescriptors

MORGAN_RADIUS = 2
FP_SIZE = 2048
# Stereo is stripped upstream so chirality is off here too - it'd just be dead bits.
INCLUDE_CHIRALITY = False
COUNT_SIMULATION = False

# Twelve descriptors, each one covering something a substructure fingerprint can't see:
# size, lipophilicity, polarity, H-bonding, flexibility, rings, saturation, complexity,
# drug-likeness.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "BertzCT",
    "qed",
)

_DESCRIPTOR_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": rdMolDescriptors.CalcNumHBD,
    "NumHAcceptors": rdMolDescriptors.CalcNumHBA,
    "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": lambda mol: float(mol.GetNumHeavyAtoms()),
    "BertzCT": Descriptors.BertzCT,
    "qed": QED.qed,
}

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=FP_SIZE,
    includeChirality=INCLUDE_CHIRALITY,
    countSimulation=COUNT_SIMULATION,
)


@dataclass(frozen=True)
class FeatureBlock:
    """A built feature matrix plus the names and block boundaries needed to read it."""

    n_molecules: int
    n_features: int
    n_fingerprint_bits: int
    n_descriptors: int
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict:
        """Return the feature-block shape metadata as a JSON-serialisable dict."""
        return {
            "n_molecules": self.n_molecules,
            "n_features": self.n_features,
            "n_fingerprint_bits": self.n_fingerprint_bits,
            "n_descriptors": self.n_descriptors,
            "feature_names_head": list(self.feature_names[:4]),
            "feature_names_tail": list(self.feature_names[-4:]),
        }


def make_morgan_generator(
    radius: int = MORGAN_RADIUS, fp_size: int = FP_SIZE, use_features: bool = False
):
    """Build a Morgan fingerprint generator; `use_features=True` gives FCFP-style invariants."""
    kwargs: dict[str, object] = {
        "radius": radius,
        "fpSize": fp_size,
        "includeChirality": INCLUDE_CHIRALITY,
        "countSimulation": COUNT_SIMULATION,
    }
    if use_features:
        kwargs["atomInvariantsGenerator"] = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
    return rdFingerprintGenerator.GetMorganGenerator(**kwargs)


def mol_from_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string, raising rather than returning None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r}")
    return mol


def fingerprint_matrix(smiles: list[str], generator=None) -> np.ndarray:
    """Return an (n, fp_size) uint8 binary ECFP matrix for a list of SMILES."""
    gen = generator or _MORGAN_GEN
    rows = [gen.GetFingerprintAsNumPy(mol_from_smiles(s)).astype(np.uint8) for s in smiles]
    return np.vstack(rows) if rows else np.zeros((0, FP_SIZE), dtype=np.uint8)


def bitvects(smiles: list[str], generator=None) -> list:
    """Return RDKit ExplicitBitVects, the representation BulkTanimotoSimilarity needs."""
    gen = generator or _MORGAN_GEN
    return [gen.GetFingerprint(mol_from_smiles(s)) for s in smiles]


def descriptor_matrix(smiles: list[str]) -> np.ndarray:
    """Return an (n, 12) float64 descriptor matrix in DESCRIPTOR_NAMES order."""
    rows = []
    for text in smiles:
        mol = mol_from_smiles(text)
        rows.append([float(_DESCRIPTOR_FUNCS[name](mol)) for name in DESCRIPTOR_NAMES])
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, len(DESCRIPTOR_NAMES)))


def build_features(
    smiles: list[str], include_descriptors: bool = True, generator=None
) -> tuple[np.ndarray, FeatureBlock]:
    """Build the model feature matrix (fingerprint bits, then descriptors) and its metadata."""
    fingerprints = fingerprint_matrix(smiles, generator=generator)
    names: list[str] = [f"fp_{i}" for i in range(fingerprints.shape[1])]

    if include_descriptors:
        descriptors = descriptor_matrix(smiles)
        matrix = np.hstack([fingerprints.astype(np.float64), descriptors])
        names.extend(DESCRIPTOR_NAMES)
        n_descriptors = descriptors.shape[1]
    else:
        matrix = fingerprints.astype(np.float64)
        n_descriptors = 0

    block = FeatureBlock(
        n_molecules=matrix.shape[0],
        n_features=matrix.shape[1],
        n_fingerprint_bits=fingerprints.shape[1],
        n_descriptors=n_descriptors,
        feature_names=tuple(names),
    )
    return matrix, block


def max_similarity_to_reference(
    query_smiles: list[str], reference_smiles: list[str], generator=None
) -> tuple[np.ndarray, np.ndarray]:
    """For each query molecule return its max ECFP4 Tanimoto to the reference set and its index."""
    query_fps = bitvects(query_smiles, generator=generator)
    reference_fps = bitvects(reference_smiles, generator=generator)
    if not reference_fps:
        raise ValueError("max_similarity_to_reference called with an empty reference set.")

    max_sims = np.empty(len(query_fps), dtype=np.float64)
    argmax = np.empty(len(query_fps), dtype=np.int64)
    for i, fingerprint in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprint, reference_fps))
        index = int(sims.argmax())
        max_sims[i] = sims[index]
        argmax[i] = index
    return max_sims, argmax
