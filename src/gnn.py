"""Molecular graphs, a GINE network, and the training loop.

To be clear about what this is, since the README makes a claim about it: it's GINEConv,
not a re-implementation of the Gilmer 2017 NNConv MPNN. Same message-passing family, and
it does use the bond features, but it projects edge features to the node dimension and
adds them instead of learning a dense edge_dim x hidden x hidden transform. At hidden=128
that transform is a 16,384-output MLP per layer, which on a few thousand molecules would
just overfit - and then a negative result would be about my architecture choice rather
than about the data. I'd rather say "GINE, and here's why" than claim Gilmer and quietly
build something else.

Generic - takes SMILES, labels, hyperparameters. Nothing in here knows about HDAC1.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

__all__ = [
    "GNNError",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "GNNConfig",
    "TrainResult",
    "set_global_seed",
    "mol_to_graph",
    "smiles_to_graphs",
    "GINERegressor",
    "train_model",
    "predict",
]

RDLogger.DisableLog("rdApp.*")


class GNNError(RuntimeError):
    """Raised when a molecule cannot be converted to a graph or training is misconfigured."""


# Feature vocabularies. Anything unrecognised goes in an "other" bucket instead of
# raising, but the caller gets the count so it can gate on it.

ATOM_SYMBOLS: tuple[str, ...] = (
    "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I",
)  # + "other"
DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: tuple[int, ...] = (-2, -1, 0, 1, 2)
NUM_HS: tuple[int, ...] = (0, 1, 2, 3, 4)
HYBRIDIZATIONS: tuple = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
BOND_TYPES: tuple = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
BOND_STEREOS: tuple = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOANY,
)

# 13 symbol + 6 degree + 5 charge + 5 numH + 5 hybridisation + aromatic + ring + mass
ATOM_FEATURE_DIM = (len(ATOM_SYMBOLS) + 1) + len(DEGREES) + len(FORMAL_CHARGES) + len(NUM_HS) + len(
    HYBRIDIZATIONS
) + 3
# 4 bond type + conjugated + ring + 4 stereo
BOND_FEATURE_DIM = len(BOND_TYPES) + 2 + len(BOND_STEREOS)

# No chirality feature - stereo is stripped during standardisation, so it'd be a
# constant zero block and would misrepresent what the model actually gets.


def _one_hot(value, vocabulary: Sequence, allow_other: bool = False) -> list[float]:
    """One-hot encode ``value``; with ``allow_other`` an extra trailing slot absorbs OOV."""
    size = len(vocabulary) + (1 if allow_other else 0)
    vec = [0.0] * size
    try:
        vec[list(vocabulary).index(value)] = 1.0
    except ValueError:
        if allow_other:
            vec[-1] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> tuple[list[float], bool]:
    """Feature vector for one atom, plus whether it fell into an out-of-vocabulary slot."""
    symbol = atom.GetSymbol()
    oov = symbol not in ATOM_SYMBOLS
    feats = (
        _one_hot(symbol, ATOM_SYMBOLS, allow_other=True)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetMass() / 100.0,
        ]
    )
    return feats, oov


def _bond_features(bond: Chem.Bond) -> list[float]:
    """Feature vector for one bond."""
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREOS)
    )


def mol_to_graph(mol: Chem.Mol, y: float | None = None, idx: int | None = None) -> tuple[Data, int]:
    """RDKit molecule to a PyG Data object, plus the out-of-vocabulary atom count.

    Edges go in both directions with the bond features duplicated. No self-loops -
    GINEConv already has a (1 + eps) * x self term, so adding them double-counts.
    """
    if mol is None:
        raise GNNError("cannot build a graph from a None molecule")
    if mol.GetNumAtoms() == 0:
        raise GNNError("cannot build a graph from a molecule with zero atoms")

    atom_feats, n_oov = [], 0
    for atom in mol.GetAtoms():
        feats, oov = _atom_features(atom)
        atom_feats.append(feats)
        n_oov += int(oov)
    x = torch.tensor(atom_feats, dtype=torch.float)

    src, dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        src += [i, j]
        dst += [j, i]
        edge_feats += [bf, bf]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        # A bondless fragment is still a valid graph; PyG needs correctly shaped empties.
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float)
    if idx is not None:
        data.idx = torch.tensor([int(idx)], dtype=torch.long)
    return data, n_oov


def smiles_to_graphs(
    smiles: Sequence[str], y: Sequence[float] | None = None, strict: bool = True
) -> tuple[list[Data], dict]:
    """SMILES to PyG graphs, with a stats dict.

    stats has the out-of-vocabulary atom count and fraction so the caller can gate on it.
    Otherwise you'd never notice a tenth of your atoms falling into the "other" bucket.
    """
    graphs: list[Data] = []
    n_oov = n_atoms = n_failed = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_failed += 1
            if strict:
                raise GNNError(f"SMILES failed to parse at index {i}: {smi!r}")
            continue
        g, oov = mol_to_graph(mol, None if y is None else float(y[i]), idx=i)
        graphs.append(g)
        n_oov += oov
        n_atoms += mol.GetNumAtoms()
    stats = {
        "n_graphs": len(graphs),
        "n_failed": n_failed,
        "n_atoms": n_atoms,
        "n_oov_atoms": n_oov,
        "oov_fraction": (n_oov / n_atoms) if n_atoms else 0.0,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
    }
    return graphs, stats


# --- determinism ---


def set_global_seed(seed: int) -> None:
    """Seed everything that has an RNG.

    PYTHONHASHSEED really needs to be set before the interpreter starts; setting it here
    at least means child processes inherit it. PyG's CPU scatter ops are deterministic so
    use_deterministic_algorithms(True) shouldn't raise. If it ever does, replace the op -
    don't turn the flag off.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU, kept so the helper is portable
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding (unused at num_workers=0, kept for portability)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- model ---


@dataclass(frozen=True)
class GNNConfig:
    """Every hyperparameter of the network and its training loop."""

    hidden: int = 128
    depth: int = 4
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 40
    lr_factor: float = 0.5
    lr_patience: int = 10
    min_lr: float = 1e-5
    grad_clip: float = 5.0
    huber_beta: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


class GINERegressor(nn.Module):
    """GINE regressor, residual connections, mean+sum readout.

    Both poolings because sum carries molecular size (which correlates with potency here)
    and mean doesn't. Giving the head both lets it pick rather than me guessing.
    """

    def __init__(self, cfg: GNNConfig, atom_dim: int = ATOM_FEATURE_DIM, bond_dim: int = BOND_FEATURE_DIM):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.atom_encoder = nn.Linear(atom_dim, h)
        self.bond_encoder = nn.Linear(bond_dim, h)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(cfg.depth):
            mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
            self.convs.append(GINEConv(mlp, edge_dim=h))
            self.norms.append(nn.BatchNorm1d(h))
        self.dropout = nn.Dropout(cfg.dropout)

        self.head = nn.Sequential(
            nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

    def forward(self, data) -> torch.Tensor:
        """Predict a standardised target for each graph in the batch."""
        x = self.atom_encoder(data.x)
        e = self.bond_encoder(data.edge_attr)
        for conv, norm in zip(self.convs, self.norms):
            # Residual around each message-passing block keeps depth 4 trainable.
            x = x + self.dropout(torch.relu(norm(conv(x, data.edge_index, e))))
        pooled = torch.cat(
            [global_mean_pool(x, data.batch), global_add_pool(x, data.batch)], dim=1
        )
        return self.head(pooled).squeeze(-1)

    def n_parameters(self) -> int:
        """Total trainable parameter count, reported next to the training-set size."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- training ---


@dataclass
class TrainResult:
    """Outcome of one training run, including the curve needed to justify early stopping."""

    best_epoch: int
    best_val_rmse: float
    epochs_run: int
    n_parameters: int
    train_losses: list[float] = field(default_factory=list)
    val_rmses: list[float] = field(default_factory=list)
    target_mean: float = 0.0
    target_std: float = 1.0
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _rmse_native(model: nn.Module, loader: DataLoader, mean: float, std: float, device) -> tuple[float, np.ndarray]:
    """Validation RMSE in native pIC50 units, with predictions de-standardised."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append((model(batch) * std + mean).cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    p = np.concatenate(preds)
    t = np.concatenate(trues)
    return float(np.sqrt(np.mean((p - t) ** 2))), p


def train_model(
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data] | None,
    cfg: GNNConfig,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
    fixed_epochs: int | None = None,
) -> tuple[GINERegressor, TrainResult]:
    """Train one GINE model, with early stopping or for a fixed number of epochs.

    The target is z-scored on the fit fold only and predictions are converted back to
    pIC50 before anything is scored, so the scaler never sees the test set.

    fixed_epochs is for the final refit: once you've refit on train+val there's no
    held-out fold left to stop on, so it runs for however many epochs early stopping
    picked during tuning. The forest gets refit on train+val too - doing it for one model
    and not the other would quietly tilt the comparison.
    """
    if not train_graphs:
        raise GNNError("train_graphs is empty")
    if fixed_epochs is None and not val_graphs:
        raise GNNError("val_graphs is empty - early stopping needs a validation fold")
    if fixed_epochs is not None and fixed_epochs < 1:
        raise GNNError(f"fixed_epochs must be >= 1, got {fixed_epochs}")

    set_global_seed(seed)
    device = torch.device(device)

    y_train = np.array([float(g.y.item()) for g in train_graphs])
    mean, std = float(y_train.mean()), float(y_train.std())
    if std == 0:
        raise GNNError("training targets are constant; cannot standardise")

    # Standardise in place on copies of the label tensors, leaving the originals alone.
    train_std = []
    for g in train_graphs:
        g2 = g.clone()
        g2.y_raw = g.y.clone()
        g2.y = (g.y - mean) / std
        train_std.append(g2)

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_std, batch_size=cfg.batch_size, shuffle=True,
        generator=generator, worker_init_fn=_seed_worker, num_workers=0,
    )
    # Validation keeps native-unit labels so RMSE is directly comparable to the table.
    val_loader = (
        DataLoader(list(val_graphs), batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        if val_graphs
        else None
    )

    model = GINERegressor(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, min_lr=cfg.min_lr
    )
    # Huber, not MSE. Some label noise gets through the pIC50-range filter and MSE lets
    # a few bad labels dominate the gradient.
    criterion = nn.SmoothL1Loss(beta=cfg.huber_beta)

    result = TrainResult(
        best_epoch=-1, best_val_rmse=float("inf"), epochs_run=0,
        n_parameters=model.n_parameters(), target_mean=mean, target_std=std, seed=seed,
    )
    best_state = None
    since_improved = 0

    n_epochs = fixed_epochs if fixed_epochs is not None else cfg.max_epochs
    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch), batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += float(loss.item()) * batch.num_graphs
            n_seen += batch.num_graphs

        result.train_losses.append(epoch_loss / max(n_seen, 1))
        result.epochs_run = epoch

        if val_loader is None:
            # Fixed-epoch refit - nothing held out to schedule or stop on, so just run
            # the epochs we were told to.
            continue

        val_rmse, _ = _rmse_native(model, val_loader, mean, std, device)
        scheduler.step(val_rmse)
        result.val_rmses.append(val_rmse)

        if val_rmse < result.best_val_rmse - 1e-6:
            result.best_val_rmse = val_rmse
            result.best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                if verbose:
                    print(f"      early stop at epoch {epoch} (best {result.best_epoch})")
                break
        if verbose and epoch % 20 == 0:
            print(f"      epoch {epoch:3d}  train {result.train_losses[-1]:.4f}  val RMSE {val_rmse:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    if fixed_epochs is not None:
        result.best_epoch = n_epochs
    return model, result


def predict(
    model: GINERegressor, graphs: Sequence[Data], result: TrainResult,
    batch_size: int = 256, device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict pIC50 in native units, inverting the train-fold target standardisation."""
    device = torch.device(device)
    model.eval().to(device)
    loader = DataLoader(list(graphs), batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out.append((model(batch) * result.target_std + result.target_mean).cpu().numpy())
    return np.concatenate(out)
