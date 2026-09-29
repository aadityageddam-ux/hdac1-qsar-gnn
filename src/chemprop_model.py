"""Chemprop D-MPNN wrapper: datapoints, model, training, prediction.

Generic - takes frames of SMILES and labels, knows nothing about a particular target.
Everything that trains a D-MPNN imports from here, so the headline run, the learning
curve and the multi-target replication all use the same model code rather than three
copies that could drift.

Chemprop 2.x at the authors' defaults: bond-level message passing, mean aggregation,
regression FFN. The defaults are what the field benchmarks against, which is the point -
this is meant to be a reference, not a tuned competitor.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl

from chemprop import data as cp_data
from chemprop import featurizers, models, nn

__all__ = [
    "MAX_EPOCHS",
    "PATIENCE",
    "BATCH_SIZE",
    "EpochRecorder",
    "make_datapoints",
    "build_model",
    "train_once",
    "predict",
]

MAX_EPOCHS = 50          # Chemprop's own default
PATIENCE = 15
BATCH_SIZE = 64


class EpochRecorder(pl.callbacks.Callback):
    """Record validation loss each epoch so the best epoch can be selected explicitly."""

    def __init__(self) -> None:
        self.val_losses: list[float] = []

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        value = trainer.callback_metrics.get("val_loss")
        if value is not None and not trainer.sanity_checking:
            self.val_losses.append(float(value))


def make_datapoints(frame: pd.DataFrame) -> list:
    """Build Chemprop datapoints from a split frame."""
    return [
        cp_data.MoleculeDatapoint.from_smi(smi, np.array([y], dtype=float))
        for smi, y in zip(frame.canonical_smiles, frame.pIC50)
    ]


def build_model() -> models.MPNN:
    """Chemprop D-MPNN at author defaults: bond-level message passing, mean aggregation."""
    return models.MPNN(
        nn.BondMessagePassing(),
        nn.MeanAggregation(),
        nn.RegressionFFN(),
        batch_norm=True,
        metrics=[nn.metrics.RMSE()],
    )


def train_once(
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame | None,
    seed: int,
    max_epochs: int,
    use_early_stopping: bool,
) -> tuple[models.MPNN, cp_data.MoleculeDataset, list[float], int]:
    """Train one D-MPNN. Returns the model, the scaled dataset and the val curve.

    Target scaling is fit on the fit fold only and then applied to val, so the scaler
    never sees anything the model gets scored on.
    """
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    fit_ds = cp_data.MoleculeDataset(make_datapoints(fit_frame), featurizer)
    scaler = fit_ds.normalize_targets()

    fit_dl = cp_data.build_dataloader(
        fit_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, seed=seed
    )

    val_dl = None
    if val_frame is not None:
        val_ds = cp_data.MoleculeDataset(make_datapoints(val_frame), featurizer)
        val_ds.normalize_targets(scaler)
        val_dl = cp_data.build_dataloader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)

    model = build_model()
    model.predictor.output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)

    recorder = EpochRecorder()
    callbacks: list = [recorder]
    if use_early_stopping and val_dl is not None:
        callbacks.append(
            pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE)
        )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )
    trainer.fit(model, fit_dl, val_dl)

    best_epoch = int(np.argmin(recorder.val_losses)) + 1 if recorder.val_losses else max_epochs
    return model, fit_ds, recorder.val_losses, best_epoch


def predict(model: models.MPNN, frame: pd.DataFrame) -> np.ndarray:
    """Predict pIC50 in native units for a frame of SMILES."""
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    ds = cp_data.MoleculeDataset(make_datapoints(frame), featurizer)
    dl = cp_data.build_dataloader(ds, batch_size=256, num_workers=0, shuffle=False)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=False,
        enable_progress_bar=False, enable_model_summary=False,
    )
    return torch.cat(trainer.predict(model, dl)).numpy().ravel()
