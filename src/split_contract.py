"""The one loader for the split, checked against a recorded SHA-256.

This module is the whole leak-free claim, and you can check it without rerunning
anything. Model code imports load_split() and nothing else - none of them may import
scaffold_split, so none of them can recompute, reorder or quietly redefine the folds.

Test-fold access goes through SplitBundle.test(), which counts calls so a gate can
assert the test set was read exactly once.

Generic - takes a path and column names, not a target.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

SPLIT_CSV_NAME = "split_assignment.csv"
SPLIT_MANIFEST_NAME = "split_manifest.json"
SHA256_MANIFEST_KEY = "split_assignment_sha256"

REQUIRED_COLUMNS: tuple[str, ...] = (
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

FOLD_NAMES: tuple[str, str, str] = ("train", "val", "test")

# Hashed in binary chunks so the digest doesn't depend on line endings or text decoding.
HASH_CHUNK_BYTES = 1 << 20


class SplitContractError(RuntimeError):
    """Raised when the split artifact is missing, malformed, or fails its SHA-256 check."""


@dataclass(frozen=True)
class SplitBundle:
    """The verified split: train and val frames, a counted test accessor, and the digest."""

    train: pd.DataFrame = field(repr=False)
    val: pd.DataFrame = field(repr=False)
    sha256: str
    csv_path: str
    n_compounds: int
    fold_sizes: dict[str, int]
    _test: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    _test_accesses: list[int] = field(repr=False, default_factory=lambda: [0])

    def test(self) -> pd.DataFrame:
        """Return the test fold and record that it was accessed."""
        self._test_accesses[0] += 1
        return self._test

    @property
    def n_test_accesses(self) -> int:
        """Number of times the test fold has been read through this bundle."""
        return self._test_accesses[0]

    def to_dict(self) -> dict:
        """Return the contract metadata (not the frames) as a JSON-serialisable dict."""
        payload = asdict(self)
        for key in ("train", "val", "_test", "_test_accesses"):
            payload.pop(key, None)
        payload["n_test_accesses"] = self.n_test_accesses
        return payload


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(
    split_csv: Path,
    manifest_json: Path | None = None,
    expected_sha256: str | None = None,
) -> SplitBundle:
    """Load the split CSV, verify its SHA-256, and return the one permitted split object."""
    split_csv = Path(split_csv)
    if not split_csv.exists():
        raise SplitContractError(f"Split artifact {split_csv} does not exist.")

    observed = sha256_file(split_csv)

    if expected_sha256 is None and manifest_json is not None:
        manifest_json = Path(manifest_json)
        if not manifest_json.exists():
            raise SplitContractError(f"Split manifest {manifest_json} does not exist.")
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
        if SHA256_MANIFEST_KEY not in manifest:
            raise SplitContractError(
                f"Manifest {manifest_json.name} has no {SHA256_MANIFEST_KEY!r} key."
            )
        expected_sha256 = manifest[SHA256_MANIFEST_KEY]

    if expected_sha256 is None:
        raise SplitContractError(
            "load_split needs either expected_sha256 or a manifest containing it."
        )
    if observed != expected_sha256:
        raise SplitContractError(
            f"Split SHA-256 mismatch for {split_csv.name}: "
            f"observed {observed}, manifest says {expected_sha256}. "
            "The split artifact has changed since it was built."
        )

    frame = pd.read_csv(split_csv)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SplitContractError(f"Split artifact is missing columns {missing}.")

    folds = set(frame["fold"].unique())
    if folds != set(FOLD_NAMES):
        raise SplitContractError(f"Split folds are {sorted(folds)}, expected {list(FOLD_NAMES)}.")

    subsets = {name: frame[frame["fold"] == name].reset_index(drop=True) for name in FOLD_NAMES}
    return SplitBundle(
        train=subsets["train"],
        val=subsets["val"],
        sha256=observed,
        csv_path=str(split_csv),
        n_compounds=len(frame),
        fold_sizes={name: len(subsets[name]) for name in FOLD_NAMES},
        _test=subsets["test"],
    )
