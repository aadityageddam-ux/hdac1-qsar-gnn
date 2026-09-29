"""Bemis-Murcko scaffold split. Deterministic, no RNG anywhere.

Atom-typed scaffolds, not generic frameworks. MakeScaffoldGeneric throws away atom
identity and would merge a hydroxamate-bearing aryl with a benzamide-bearing one. When
the zinc-binding group is the pharmacophore, that makes the split hard for reasons that
have nothing to do with structural novelty.

The trap is ring-free molecules: MurckoScaffoldSmiles returns "" for them, and the
obvious handling lumps unrelated compounds into one enormous pseudo-scaffold. Each
acyclic molecule gets its own singleton group instead.

Generic - takes SMILES and fractions, not a target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Stereo is stripped upstream, so scaffolds are computed achirally to match.
MURCKO_INCLUDE_CHIRALITY = False

# Key prefix that keeps every ring-free molecule in its own singleton group.
ACYCLIC_PREFIX = "ACYCLIC::"

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Fold fractions have to land within this many points of what was asked, or it raises
# rather than quietly rebalancing.
FRACTION_TOLERANCE = 0.03

# This chemical space is ring-dominated. A big acyclic fraction means standardization
# broke, not that the chemistry changed.
MAX_ACYCLIC_FRACTION = 0.05

FOLD_NAMES: tuple[str, str, str] = ("train", "val", "test")

ALGORITHM_NAME = "deterministic-greedy-murcko-descending-size"
SCAFFOLD_FUNCTION_SIGNATURE = (
    "MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)"
)


class ScaffoldSplitError(RuntimeError):
    """Raised when a scaffold split cannot be produced within its stated tolerances."""


@dataclass(frozen=True)
class SplitResult:
    """A completed scaffold split: per-compound assignment plus reproducibility metadata."""

    frame: pd.DataFrame = field(repr=False)
    n_compounds: int
    n_groups: int
    n_acyclic: int
    fold_sizes: dict[str, int]
    fold_fractions: dict[str, float]
    fold_group_counts: dict[str, int]
    requested_fractions: dict[str, float]

    def to_dict(self) -> dict:
        """Return the split metadata (not the assignment frame) as a JSON-serialisable dict."""
        return {
            "algorithm": ALGORITHM_NAME,
            "scaffold_function": SCAFFOLD_FUNCTION_SIGNATURE,
            "n_compounds": self.n_compounds,
            "n_groups": self.n_groups,
            "n_acyclic": self.n_acyclic,
            "acyclic_fraction": round(self.n_acyclic / self.n_compounds, 6),
            "fold_sizes": self.fold_sizes,
            "fold_fractions": {k: round(v, 6) for k, v in self.fold_fractions.items()},
            "fold_group_counts": self.fold_group_counts,
            "requested_fractions": self.requested_fractions,
            "fraction_tolerance": FRACTION_TOLERANCE,
        }


def murcko_scaffold(smiles: str) -> str:
    """Return the atom-typed Bemis-Murcko scaffold SMILES, or "" for a ring-free molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ScaffoldSplitError(f"Could not parse SMILES {smiles!r} for scaffold extraction.")
    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=MURCKO_INCLUDE_CHIRALITY
    )


def scaffold_keys(smiles: list[str]) -> tuple[list[str], list[str], list[bool]]:
    """Return (scaffold smiles, grouping key, is_acyclic) for each input SMILES."""
    scaffolds: list[str] = []
    keys: list[str] = []
    acyclic: list[bool] = []
    for text in smiles:
        scaffold = murcko_scaffold(text)
        is_acyclic = scaffold == ""
        scaffolds.append(scaffold)
        keys.append(f"{ACYCLIC_PREFIX}{text}" if is_acyclic else scaffold)
        acyclic.append(is_acyclic)
    return scaffolds, keys, acyclic


def build_split(
    frame: pd.DataFrame,
    smiles_column: str = "canonical_smiles",
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    tolerance: float = FRACTION_TOLERANCE,
) -> SplitResult:
    """Assign every compound to train/val/test by descending scaffold-group size, no RNG."""
    if smiles_column not in frame.columns:
        raise ScaffoldSplitError(f"Column {smiles_column!r} not in {list(frame.columns)}")
    n_compounds = len(frame)
    if n_compounds == 0:
        raise ScaffoldSplitError("build_split received an empty frame.")

    out = frame.copy().reset_index(drop=True)
    scaffolds, keys, acyclic = scaffold_keys(out[smiles_column].tolist())
    out["murcko_scaffold"] = scaffolds
    out["scaffold_key"] = keys
    out["is_acyclic"] = acyclic

    members: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        members.setdefault(key, []).append(index)

    # Biggest groups first, so train soaks up the common chemotypes and test is left with
    # the rare and singleton scaffolds. That's the harder setting and the honest one. The
    # lexicographic tiebreak keeps the order stable across runs, platforms and hash seeds.
    ordered = sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))

    n_train_target = round(n_compounds * train_fraction)
    n_trainval_target = round(n_compounds * (train_fraction + val_fraction))

    fold_of_index: list[str] = [""] * n_compounds
    group_id_of_index: list[int] = [-1] * n_compounds
    group_size_of_index: list[int] = [0] * n_compounds
    fold_group_counts = {name: 0 for name in FOLD_NAMES}

    placed = 0
    for group_id, (key, indices) in enumerate(ordered):
        if placed < n_train_target:
            fold = "train"
        elif placed < n_trainval_target:
            fold = "val"
        else:
            fold = "test"
        fold_group_counts[fold] += 1
        for index in indices:
            fold_of_index[index] = fold
            group_id_of_index[index] = group_id
            group_size_of_index[index] = len(indices)
        placed += len(indices)

    out["fold"] = fold_of_index
    out["scaffold_group_id"] = group_id_of_index
    out["scaffold_group_size"] = group_size_of_index

    fold_sizes = {name: int((out["fold"] == name).sum()) for name in FOLD_NAMES}
    fold_fractions = {name: fold_sizes[name] / n_compounds for name in FOLD_NAMES}
    requested = {
        "train": train_fraction,
        "val": val_fraction,
        "test": round(1.0 - train_fraction - val_fraction, 10),
    }

    for name in FOLD_NAMES:
        if fold_sizes[name] == 0:
            raise ScaffoldSplitError(f"Fold {name!r} is empty.")
        deviation = abs(fold_fractions[name] - requested[name])
        if deviation > tolerance:
            raise ScaffoldSplitError(
                f"Fold {name!r} fraction {fold_fractions[name]:.4f} deviates "
                f"{deviation:.4f} from requested {requested[name]:.4f} "
                f"(tolerance {tolerance})."
            )

    n_acyclic = int(out["is_acyclic"].sum())
    if n_acyclic / n_compounds >= MAX_ACYCLIC_FRACTION:
        raise ScaffoldSplitError(
            f"Acyclic fraction {n_acyclic / n_compounds:.4f} >= {MAX_ACYCLIC_FRACTION}; "
            "standardization has probably fragmented the structures."
        )

    return SplitResult(
        frame=out,
        n_compounds=n_compounds,
        n_groups=len(ordered),
        n_acyclic=n_acyclic,
        fold_sizes=fold_sizes,
        fold_fractions=fold_fractions,
        fold_group_counts=fold_group_counts,
        requested_fractions=requested,
    )
