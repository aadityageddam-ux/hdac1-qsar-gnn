"""Random-forest baseline: tuning on the validation fold, multi-seed fitting, ablations.

Why a random forest rather than gradient boosting, since the README makes this claim:

* A few thousand training rows against ~2,060 mostly-sparse binary features is where
  bagging's variance reduction beats boosting's bias reduction.
* A forest reaches its ceiling with almost no tuning. That matters for credibility: the
  easiest way to dismiss a "the GNN did not win" result is to argue the baseline was
  under-tuned, and boosting would need learning rate, tree count, depth and subsample
  tuned jointly before that objection could be answered.
* ``HistGradientBoostingRegressor`` bins each feature into 255 buckets, which is
  pointless on binary inputs, and would densify 2048 columns.

No feature scaling anywhere. A forest is invariant to any monotone per-feature
transform, so a ``StandardScaler`` would change nothing except introduce a fitted
object that could leak across folds. The real mixed-feature problem is that 12
descriptors compete with 2048 bits for split candidates, which is handled by putting
``max_features`` in the tuning grid and by reporting the feature-block ablations.

This module never touches the test fold. Test predictions are produced by the caller
passing in an already-extracted matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import evaluate as ev

__all__ = [
    "BaselineError",
    "RFConfig",
    "GridRow",
    "TuningResult",
    "MultiSeedResult",
    "DEFAULT_GRID",
    "SEEDS",
    "tune_random_forest",
    "fit_predict_multiseed",
    "split_importance",
]


class BaselineError(RuntimeError):
    """Raised when the baseline is given inconsistent data or an empty grid."""


# n_estimators is fixed rather than tuned: random-forest test error is monotone
# non-increasing in the number of trees, so "tuning" it only ever selects the largest
# value offered. Take the largest that is affordable and spend the search elsewhere.
N_ESTIMATORS = 1000
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

DEFAULT_GRID: tuple[dict, ...] = tuple(
    {"max_features": mf, "min_samples_leaf": msl}
    for mf in ("sqrt", 0.1, 0.3)
    for msl in (1, 2, 5)
)


@dataclass(frozen=True)
class RFConfig:
    """One random-forest configuration."""

    max_features: object = "sqrt"
    min_samples_leaf: int = 1
    n_estimators: int = N_ESTIMATORS
    bootstrap: bool = True
    n_jobs: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    def build(self, random_state: int) -> RandomForestRegressor:
        """Instantiate the estimator for a given seed."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=random_state,
        )


@dataclass(frozen=True)
class GridRow:
    """One point of the validation-fold search."""

    config: dict
    val_rmse: float
    val_spearman: float
    fit_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TuningResult:
    """Outcome of the validation-fold grid search."""

    best_config: dict
    best_val_rmse: float
    best_val_spearman: float
    rows: list[GridRow] = field(default_factory=list)
    selection_metric: str = "val_rmse (ties broken on val_spearman)"
    tuned_on: str = "validation fold only; the test fold is not read by this module"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["rows"] = [r.to_dict() for r in self.rows]
        return payload


@dataclass
class MultiSeedResult:
    """Predictions and importances across seeds, plus the mean prediction used for scoring."""

    seeds: list[int]
    mean_prediction: np.ndarray = field(repr=False)
    per_seed_predictions: np.ndarray = field(repr=False)
    per_seed_rmse: list[float] = field(default_factory=list)
    feature_importances: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))
    fit_seconds: float = 0.0

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)

    def to_dict(self) -> dict:
        """Serialisable summary; the prediction arrays are written separately as .npy."""
        return {
            "seeds": self.seeds,
            "n_seeds": self.n_seeds,
            "per_seed_rmse": self.per_seed_rmse,
            "per_seed_rmse_mean": float(np.mean(self.per_seed_rmse)) if self.per_seed_rmse else None,
            "per_seed_rmse_sd": float(np.std(self.per_seed_rmse, ddof=1))
            if len(self.per_seed_rmse) > 1
            else 0.0,
            "fit_seconds": self.fit_seconds,
        }


def _check(X: np.ndarray, y: np.ndarray, name: str) -> None:
    """Reject malformed matrices loudly rather than letting sklearn guess."""
    if X.ndim != 2:
        raise BaselineError(f"{name}: expected a 2-D matrix, got shape {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise BaselineError(f"{name}: {X.shape[0]} rows but {y.shape[0]} labels")
    if X.shape[0] == 0:
        raise BaselineError(f"{name}: empty matrix")
    if not np.isfinite(X).all():
        raise BaselineError(f"{name}: matrix contains non-finite values")


def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    grid: Iterable[dict] = DEFAULT_GRID,
    seed: int = 0,
    verbose: bool = True,
) -> TuningResult:
    """Grid-search on the validation fold only, as an explicit loop.

    Written as a visible loop rather than ``GridSearchCV`` so that the fold discipline
    is auditable in the source: the test fold is never passed to this function, and the
    selection criterion is applied to validation predictions alone.
    """
    import time

    _check(X_train, y_train, "train")
    _check(X_val, y_val, "val")
    grid = list(grid)
    if not grid:
        raise BaselineError("empty hyperparameter grid")

    rows: list[GridRow] = []
    for i, params in enumerate(grid, 1):
        cfg = RFConfig(**params)
        t0 = time.time()
        model = cfg.build(random_state=seed).fit(X_train, y_train)
        pred = model.predict(X_val)
        elapsed = time.time() - t0
        row = GridRow(
            config=cfg.to_dict(),
            val_rmse=ev.compute_metric("rmse", y_val, pred),
            val_spearman=ev.compute_metric("spearman", y_val, pred),
            fit_seconds=elapsed,
        )
        rows.append(row)
        if verbose:
            print(
                f"    [{i}/{len(grid)}] max_features={params['max_features']!s:>5} "
                f"min_samples_leaf={params['min_samples_leaf']}  "
                f"val RMSE {row.val_rmse:.4f}  val rho {row.val_spearman:.4f}  ({elapsed:.1f}s)"
            )

    # Lower RMSE wins; ties broken by higher Spearman, since ranking is the practical task.
    best = min(rows, key=lambda r: (round(r.val_rmse, 6), -r.val_spearman))
    return TuningResult(
        best_config=best.config,
        best_val_rmse=best.val_rmse,
        best_val_spearman=best.val_spearman,
        rows=rows,
    )


def fit_predict_multiseed(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_eval: np.ndarray,
    config: dict,
    y_eval: np.ndarray | None = None,
    seeds: Sequence[int] = SEEDS,
    verbose: bool = True,
) -> MultiSeedResult:
    """Fit one configuration under several seeds and average the predictions.

    The GNN is reported over five seeds because small-data graph models have real seed
    variance; running the forest over the same five keeps the comparison symmetric.
    Reporting a single-seed forest against a five-seed GNN mean would quietly rig the
    result, so the seed list is shared rather than passed in per model.
    """
    import time

    _check(X_fit, y_fit, "fit")
    if X_eval.shape[1] != X_fit.shape[1]:
        raise BaselineError(
            f"eval matrix has {X_eval.shape[1]} features but fit matrix has {X_fit.shape[1]}"
        )

    preds, rmses, importances = [], [], []
    t0 = time.time()
    for seed in seeds:
        model = RFConfig(**config).build(random_state=seed).fit(X_fit, y_fit)
        pred = model.predict(X_eval)
        preds.append(pred)
        importances.append(model.feature_importances_)
        if y_eval is not None:
            rmses.append(ev.compute_metric("rmse", y_eval, pred))
        if verbose:
            tail = f"  RMSE {rmses[-1]:.4f}" if rmses else ""
            print(f"    seed {seed}: fitted{tail}")
    elapsed = time.time() - t0

    per_seed = np.vstack(preds)
    return MultiSeedResult(
        seeds=list(seeds),
        mean_prediction=per_seed.mean(axis=0),
        per_seed_predictions=per_seed,
        per_seed_rmse=rmses,
        feature_importances=np.vstack(importances).mean(axis=0),
        fit_seconds=elapsed,
    )


def split_importance(
    importances: np.ndarray, n_fingerprint_bits: int, descriptor_names: Sequence[str]
) -> dict:
    """Divide total Gini importance between fingerprint bits and descriptors.

    Relevant to how the headline comparison should be read: if the twelve bulk
    descriptors carry most of the importance, then "random forest versus GNN" is really
    "bulk properties versus substructure", and the README has to say so.
    """
    imp = np.asarray(importances, dtype=np.float64)
    if imp.size != n_fingerprint_bits + len(descriptor_names):
        raise BaselineError(
            f"importance vector has {imp.size} entries but expected "
            f"{n_fingerprint_bits} bits + {len(descriptor_names)} descriptors"
        )
    fp_part = imp[:n_fingerprint_bits]
    desc_part = imp[n_fingerprint_bits:]
    order = np.argsort(fp_part)[::-1][:20]
    return {
        "fingerprint_total": float(fp_part.sum()),
        "descriptor_total": float(desc_part.sum()),
        "fingerprint_mean_per_feature": float(fp_part.mean()),
        "descriptor_mean_per_feature": float(desc_part.mean()) if desc_part.size else 0.0,
        "descriptors": {
            name: float(value) for name, value in zip(descriptor_names, desc_part)
        },
        "top_fingerprint_bits": [
            {"bit": int(b), "importance": float(fp_part[b])} for b in order
        ],
    }
