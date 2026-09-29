"""Every metric in the project is computed here.

The models just hand back a prediction vector; this file does the scoring. Keeping it
in one place is the only reason I can claim the models were measured the same way.

Two things worth knowing before you read on:

- bootstrap_indices caches one resample matrix per test-set size, so both models get
  scored on the same resamples. paired_bootstrap_delta needs that to be a real paired
  test rather than two separate ones.
- The baselines at the bottom return predictions, not metrics, so they go through the
  same code as the real models. No special cases.
"""

from __future__ import annotations

import ast
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)

__all__ = [
    "EvaluationError",
    "RegressionMetrics",
    "ClassificationMetrics",
    "BootstrapCI",
    "DeltaResult",
    "N_BOOTSTRAP",
    "BOOTSTRAP_SEED",
    "THRESHOLD_PRIMARY",
    "THRESHOLDS",
    "METRIC_NAMES",
    "HIGHER_IS_BETTER",
    "bootstrap_indices",
    "compute_metric",
    "regression_metrics",
    "classification_metrics",
    "bootstrap_ci",
    "paired_bootstrap_delta",
    "mean_predictor",
    "majority_class_predictor",
    "random_predictor",
    "one_nn_tanimoto_predict",
    "module_sha256",
    "verdict",
    "load_verdict_from_json",
]


class EvaluationError(RuntimeError):
    """Bad input to the scoring code."""


# Protocol constants, not tuning knobs. Changing one invalidates a comparison.

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260925

# 100 nM is the standard potency cut in med chem. I picked it before scoring anything,
# and not because of class balance. The other two get reported alongside so it's clear
# I didn't go looking for the threshold that flattered the result.
THRESHOLD_PRIMARY = 7.0
THRESHOLDS: tuple[float, ...] = (7.0, 6.5, 6.0)

# Percentile bootstrap rather than BCa. BCa's acceleration term is unstable for AUC,
# and I'd rather ship something simple that I'm sure is right.
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

REGRESSION_METRICS: tuple[str, ...] = ("rmse", "mae", "r2", "spearman", "pearson")
CLASSIFICATION_METRICS: tuple[str, ...] = ("roc_auc", "pr_auc", "mcc", "balanced_accuracy")
METRIC_NAMES: tuple[str, ...] = REGRESSION_METRICS + CLASSIFICATION_METRICS

HIGHER_IS_BETTER: dict[str, bool] = {
    "rmse": False,
    "mae": False,
    "r2": True,
    "spearman": True,
    "pearson": True,
    "roc_auc": True,
    "pr_auc": True,
    "mcc": True,
    "balanced_accuracy": True,
}

# Dimensionless metrics, so the verdict text doesn't call them log units.
_UNITLESS = frozenset(
    {"r2", "spearman", "pearson", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"}
)


# --- result containers ---


@dataclass(frozen=True)
class RegressionMetrics:
    """Point-estimate regression metrics on one prediction vector."""

    rmse: float
    mae: float
    r2: float
    spearman: float
    pearson: float
    n: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification metrics from thresholding the predicted pIC50.

    There's no separate classifier - the regressor's output is the score, so the
    regression and classification numbers describe the same model.
    """

    threshold: float
    roc_auc: float
    pr_auc: float
    mcc: float
    balanced_accuracy: float
    n_pos: int
    n_neg: int
    pos_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap interval for one metric on one model."""

    metric: str
    point: float
    lo: float
    hi: float
    n_boot: int
    n_valid: int
    n_skipped: int
    method: str = "percentile"

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"


@dataclass(frozen=True)
class DeltaResult:
    """Paired bootstrap on the difference between two models.

    delta is always metric_a - metric_b. favours uses HIGHER_IS_BETTER so it reads
    correctly for RMSE (lower wins) and AUC (higher wins) alike.
    """

    metric: str
    point_a: float
    point_b: float
    delta: float
    lo: float
    hi: float
    p_value: float
    significant: bool
    favours: Literal["a", "b", "neither"]
    n_boot: int
    n_valid: int
    label_a: str = "a"
    label_b: str = "b"

    def to_dict(self) -> dict:
        return asdict(self)


# --- shared resample matrix ---


@lru_cache(maxsize=None)
def bootstrap_indices(
    n: int, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Cached (n_boot, n) resample index matrix for a test set of size n.

    Cached so every metric and every model sees the same resamples - otherwise the
    paired test isn't paired.
    """
    if n <= 0:
        raise EvaluationError(f"bootstrap_indices needs a positive test-set size, got {n}")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int64)


# --- metrics ---
# They all take (y_true, y_pred, threshold) so the bootstrap can loop over them.


def _rmse(y_true, y_pred, _t=None) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _mae(y_true, y_pred, _t=None) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true, y_pred, _t=None) -> float:
    return float(r2_score(y_true, y_pred))


def _is_constant(a: np.ndarray) -> bool:
    """Is every value identical?

    ptp, not std == 0. np.std of identical floats isn't exactly zero - the mean usually
    isn't exactly representable, so you're left with ~1e-15 and the guard never fires.
    I got caught by this. ptp is max minus min, no arithmetic, so it's exactly zero
    when the values really are identical.
    """
    return a.size == 0 or float(np.ptp(a)) == 0.0


def _spearman(y_true, y_pred, _t=None) -> float:
    # No defined correlation for a constant prediction. NaN is the right answer - a model
    # that collapsed to the mean shouldn't score as if it ranked anything.
    if _is_constant(y_pred) or _is_constant(y_true):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.spearmanr(y_true, y_pred).statistic)


def _pearson(y_true, y_pred, _t=None) -> float:
    if _is_constant(y_pred) or _is_constant(y_true):
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(stats.pearsonr(y_true, y_pred).statistic)


def _roc_auc(y_true, y_pred, threshold) -> float:
    y_bin = y_true >= threshold
    if y_bin.all() or not y_bin.any():
        return float("nan")
    return float(roc_auc_score(y_bin, y_pred))


def _pr_auc(y_true, y_pred, threshold) -> float:
    # average_precision_score, not auc(recall, precision). The second one interpolates
    # across the PR curve and reads high.
    y_bin = y_true >= threshold
    if y_bin.all() or not y_bin.any():
        return float("nan")
    return float(average_precision_score(y_bin, y_pred))


def _mcc(y_true, y_pred, threshold) -> float:
    y_bin = y_true >= threshold
    if y_bin.all() or not y_bin.any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(matthews_corrcoef(y_bin, y_pred >= threshold))


def _balanced_accuracy(y_true, y_pred, threshold) -> float:
    y_bin = y_true >= threshold
    if y_bin.all() or not y_bin.any():
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(y_bin, y_pred >= threshold))


_METRIC_FNS: dict[str, Callable[..., float]] = {
    "rmse": _rmse,
    "mae": _mae,
    "r2": _r2,
    "spearman": _spearman,
    "pearson": _pearson,
    "roc_auc": _roc_auc,
    "pr_auc": _pr_auc,
    "mcc": _mcc,
    "balanced_accuracy": _balanced_accuracy,
}


def _as_arrays(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float arrays and fail loudly on bad shapes or non-finite values."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise EvaluationError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.size == 0:
        raise EvaluationError("empty evaluation arrays")
    if not np.isfinite(yt).all():
        raise EvaluationError("y_true contains non-finite values")
    if not np.isfinite(yp).all():
        raise EvaluationError("y_pred contains non-finite values")
    return yt, yp


def compute_metric(
    metric: str, y_true, y_pred, threshold: float | None = None
) -> float:
    """Compute one named metric. Classification metrics require a threshold."""
    if metric not in _METRIC_FNS:
        raise EvaluationError(f"unknown metric {metric!r}; known: {sorted(_METRIC_FNS)}")
    if metric in CLASSIFICATION_METRICS and threshold is None:
        raise EvaluationError(f"metric {metric!r} requires a threshold")
    yt, yp = _as_arrays(y_true, y_pred)
    return _METRIC_FNS[metric](yt, yp, threshold)


def regression_metrics(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> RegressionMetrics:
    """Regression metrics.

    Spearman is the one I care about most - in practice you want the ranking right more
    than you want the absolute potency right.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    return RegressionMetrics(
        rmse=_rmse(yt, yp),
        mae=_mae(yt, yp),
        r2=_r2(yt, yp),
        spearman=_spearman(yt, yp),
        pearson=_pearson(yt, yp),
        n=int(yt.size),
    )


def classification_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    threshold: float = THRESHOLD_PRIMARY,
) -> ClassificationMetrics:
    """Classification metrics at ``threshold``, scoring on the continuous prediction."""
    yt, yp = _as_arrays(y_true, y_pred)
    y_bin = yt >= threshold
    n_pos = int(y_bin.sum())
    return ClassificationMetrics(
        threshold=float(threshold),
        roc_auc=_roc_auc(yt, yp, threshold),
        pr_auc=_pr_auc(yt, yp, threshold),
        mcc=_mcc(yt, yp, threshold),
        balanced_accuracy=_balanced_accuracy(yt, yp, threshold),
        n_pos=n_pos,
        n_neg=int(y_bin.size - n_pos),
        pos_rate=float(y_bin.mean()),
    )


# --- bootstrap ---


def _bootstrap_distribution(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Metric on each resample. NaN where the resample was degenerate."""
    idx = bootstrap_indices(int(y_true.size), n_boot, seed)
    fn = _METRIC_FNS[metric]
    out = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        take = idx[i]
        out[i] = fn(y_true[take], y_pred[take], threshold)
    return out


def bootstrap_ci(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metric: str,
    threshold: float | None = None,
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapCI:
    """Percentile bootstrap CI for one metric, resampling the test set with replacement.

    Some resamples come back single-class, which leaves AUC undefined. Those get skipped
    and counted rather than filled in with something.
    """
    if metric in CLASSIFICATION_METRICS and threshold is None:
        raise EvaluationError(f"metric {metric!r} requires a threshold")
    yt, yp = _as_arrays(y_true, y_pred)
    dist = _bootstrap_distribution(metric, yt, yp, threshold, n_boot, seed)
    valid = dist[np.isfinite(dist)]
    if valid.size == 0:
        raise EvaluationError(
            f"every bootstrap resample was degenerate for metric {metric!r}"
        )
    return BootstrapCI(
        metric=metric,
        point=_METRIC_FNS[metric](yt, yp, threshold),
        lo=float(np.percentile(valid, CI_LOW_PCT)),
        hi=float(np.percentile(valid, CI_HIGH_PCT)),
        n_boot=n_boot,
        n_valid=int(valid.size),
        n_skipped=int(dist.size - valid.size),
    )


def paired_bootstrap_delta(
    y_true: Sequence[float],
    y_pred_a: Sequence[float],
    y_pred_b: Sequence[float],
    metric: str,
    threshold: float | None = None,
    label_a: str = "a",
    label_b: str = "b",
    n_boot: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> DeltaResult:
    """Paired bootstrap on metric_a - metric_b. This is the comparison that matters.

    Both models use the same cached resamples, so each one gives a matched pair and the
    difference is paired. Checking whether two separate CIs overlap is not the same test
    and understates significance.
    """
    if metric in CLASSIFICATION_METRICS and threshold is None:
        raise EvaluationError(f"metric {metric!r} requires a threshold")
    yt, ya = _as_arrays(y_true, y_pred_a)
    _, yb = _as_arrays(y_true, y_pred_b)

    dist_a = _bootstrap_distribution(metric, yt, ya, threshold, n_boot, seed)
    dist_b = _bootstrap_distribution(metric, yt, yb, threshold, n_boot, seed)
    delta = dist_a - dist_b
    valid = delta[np.isfinite(delta)]
    if valid.size == 0:
        raise EvaluationError(f"no valid paired resamples for metric {metric!r}")

    lo = float(np.percentile(valid, CI_LOW_PCT))
    hi = float(np.percentile(valid, CI_HIGH_PCT))
    # Two-sided p: twice the smaller tail either side of zero.
    p_value = min(1.0, 2.0 * min(float((valid <= 0).mean()), float((valid >= 0).mean())))

    point_a = _METRIC_FNS[metric](yt, ya, threshold)
    point_b = _METRIC_FNS[metric](yt, yb, threshold)
    significant = lo > 0 or hi < 0
    if not significant:
        favours: Literal["a", "b", "neither"] = "neither"
    else:
        a_wins = (point_a > point_b) if HIGHER_IS_BETTER[metric] else (point_a < point_b)
        favours = "a" if a_wins else "b"

    return DeltaResult(
        metric=metric,
        point_a=point_a,
        point_b=point_b,
        delta=point_a - point_b,
        lo=lo,
        hi=hi,
        p_value=p_value,
        significant=significant,
        favours=favours,
        n_boot=n_boot,
        n_valid=int(valid.size),
        label_a=label_a,
        label_b=label_b,
    )


# --- baselines ---
# These return predictions, not metrics, so they get scored by the same code as the real
# models. They're always reported, which stops a mediocre model looking good on its own.


def mean_predictor(y_train: Sequence[float], n_test: int) -> np.ndarray:
    """Constant prediction at the train mean.

    Train mean, not test mean, so R^2 lands near zero and can go slightly negative. A
    real model doesn't get to centre itself on the data it's being scored on. Its RMSE
    is what makes everyone else's R^2 mean anything.
    """
    return np.full(int(n_test), float(np.mean(np.asarray(y_train, dtype=np.float64))))


def majority_class_predictor(
    y_train: Sequence[float], n_test: int, threshold: float = THRESHOLD_PRIMARY
) -> np.ndarray:
    """Puts every test compound in the training-set majority class.

    Goes through the same thresholding as a real model, so it gives balanced accuracy
    0.50, MCC 0, and PR-AUC equal to the positive rate. That last one is why pos_rate
    sits in the same table row - PR-AUC is meaningless without it.
    """
    yt = np.asarray(y_train, dtype=np.float64)
    majority_is_positive = bool((yt >= threshold).mean() >= 0.5)
    fill = threshold + 1.0 if majority_is_positive else threshold - 1.0
    return np.full(int(n_test), fill)


def random_predictor(
    n_test: int, low: float, high: float, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Uniform random scores. ROC-AUC should come out near 0.50."""
    return np.random.default_rng(seed).uniform(low, high, size=int(n_test))


def one_nn_tanimoto_predict(
    train_fps: Sequence,
    train_y: Sequence[float],
    test_fps: Sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each test compound from its nearest training compound by ECFP4 Tanimoto.

    This is the baseline I'd actually worry about. It learns nothing - it's a lookup. If
    the forest or the GNN can't clearly beat it, then the dataset is just memorisable by
    similarity and neither model has learned any SAR.

    fps have to be RDKit ExplicitBitVect. Returns (predictions, max_similarity); the
    similarity vector gets reused by the leakage gates and the stratified analysis, so
    it comes back here instead of being recomputed twice.
    """
    from rdkit import DataStructs  # local import keeps the metric core rdkit-free

    ty = np.asarray(train_y, dtype=np.float64)
    if len(train_fps) != ty.size:
        raise EvaluationError(
            f"train_fps ({len(train_fps)}) and train_y ({ty.size}) disagree in length"
        )

    train_list = list(train_fps)
    preds = np.empty(len(test_fps), dtype=np.float64)
    max_sim = np.empty(len(test_fps), dtype=np.float64)
    for i, fp in enumerate(test_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_list))
        j = int(sims.argmax())
        preds[i] = ty[j]
        max_sim[i] = float(sims[j])
    return preds, max_sim


# --- provenance and verdict ---


def module_sha256() -> str:
    """SHA-256 of this file's logic, written into every metrics JSON.

    A gate checks all the models carry the same one, which proves they were scored by
    the same code and not by two copies that drifted apart.

    It hashes the parsed syntax tree with docstrings stripped, not the raw bytes, so
    rewording a comment doesn't invalidate results that were scored by identical logic.
    Anything that changes what the code actually does still changes the hash.
    """
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)                     and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def verdict(delta: DeltaResult) -> str:
    """Write the verdict sentence from the confidence interval.

    The README's headline comes from this function, so the prose can't drift away from
    the numbers.
    """
    unit = "" if delta.metric in _UNITLESS else " log units"
    if not delta.significant:
        return (
            f"No significant difference in {delta.metric} between {delta.label_a} and "
            f"{delta.label_b} at this sample size: difference {delta.delta:+.4f}{unit}, "
            f"95% CI [{delta.lo:+.4f}, {delta.hi:+.4f}], p = {delta.p_value:.3f}."
        )
    winner = delta.label_a if delta.favours == "a" else delta.label_b
    loser = delta.label_b if delta.favours == "a" else delta.label_a
    return (
        f"{winner} outperforms {loser} on {delta.metric}: difference "
        f"{delta.delta:+.4f}{unit}, 95% CI [{delta.lo:+.4f}, {delta.hi:+.4f}], "
        f"p = {delta.p_value:.3f}."
    )


def load_verdict_from_json(path: str | Path, metric: str = "rmse") -> str:
    """Rebuild the verdict from a written comparison.json, for the README check."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return verdict(DeltaResult(**payload["deltas"][metric]))
