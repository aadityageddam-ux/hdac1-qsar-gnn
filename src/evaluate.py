"""Shared evaluation harness: the only module in this project that computes a metric.

Both the random-forest baseline and the GNN produce a ``y_pred`` vector and nothing
else. Every number that reaches a table, a figure or the README is computed here, so
the two models cannot drift apart in how they are scored.

Three things in here are load-bearing for the honesty of the comparison:

1. ``bootstrap_indices`` draws the resample matrix ONCE per test-set size and caches
   it, so both models are scored on identical resamples. That is what makes the
   paired test in ``paired_bootstrap_delta`` valid.
2. ``paired_bootstrap_delta`` is the headline statistic. Two overlapping marginal
   confidence intervals do NOT imply the two models are indistinguishable; the CI on
   the paired difference is the correct test.
3. The reference predictors at the bottom return prediction vectors rather than
   metrics, so trivial baselines flow through exactly the same metric code as the
   real models, with no special-casing.
"""

from __future__ import annotations

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
    """Raised when inputs to the evaluation harness are malformed."""


# ---------------------------------------------------------------------------------
# Constants. These are protocol, not tuning knobs - changing one invalidates a
# comparison, so they live here rather than being passed in per call site.
# ---------------------------------------------------------------------------------

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260925

# Pre-registered before any model was scored: 100 nM is the conventional
# medicinal-chemistry potency cut. It was NOT chosen for class balance. The secondary
# thresholds are reported alongside it so there is no question of threshold-shopping.
THRESHOLD_PRIMARY = 7.0
THRESHOLDS: tuple[float, ...] = (7.0, 6.5, 6.0)

# Percentile bootstrap, not BCa: BCa's acceleration term is unstable for AUC-type
# statistics, and a plainly-stated percentile interval beats a mis-implemented BCa one.
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

# Metrics that are dimensionless ratios rather than pIC50 log units, for verdict prose.
_UNITLESS = frozenset(
    {"r2", "spearman", "pearson", "roc_auc", "pr_auc", "mcc", "balanced_accuracy"}
)


# ---------------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------------


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
    """Classification metrics derived by thresholding a continuous pIC50 prediction.

    No separate classifier is trained: the regressor's continuous output is the score,
    so the regression and classification views describe one and the same model.
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
    """Paired bootstrap on the difference between two models' metric values.

    ``delta`` is always ``metric_a - metric_b``. ``favours`` is resolved through
    ``HIGHER_IS_BETTER`` so it reads correctly for RMSE (lower is better) and for
    AUC (higher is better) alike.
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


# ---------------------------------------------------------------------------------
# The shared resample matrix
# ---------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def bootstrap_indices(
    n: int, n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Return the cached (n_boot, n) resample index matrix for a test set of size n.

    Cached so every metric, and both models, are scored on identical resamples. This is
    a precondition for ``paired_bootstrap_delta`` being a genuinely paired test rather
    than two independent ones.
    """
    if n <= 0:
        raise EvaluationError(f"bootstrap_indices needs a positive test-set size, got {n}")
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int64)


# ---------------------------------------------------------------------------------
# Metric implementations. Every metric shares the (y_true, y_pred, threshold)
# signature so the bootstrap machinery can treat them uniformly.
# ---------------------------------------------------------------------------------


def _rmse(y_true, y_pred, _t=None) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def _mae(y_true, y_pred, _t=None) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _r2(y_true, y_pred, _t=None) -> float:
    return float(r2_score(y_true, y_pred))


def _is_constant(a: np.ndarray) -> bool:
    """Exact constant-input test.

    Deliberately peak-to-peak rather than ``np.std(a) == 0``: the standard deviation of
    an array of identical floats is NOT exactly zero (the mean is generally not exactly
    representable, so the two-pass variance leaves ~1e-15 of residue), and a ``== 0``
    guard silently fails to fire. ``ptp`` is a max-minus-min with no arithmetic, so it
    is exactly zero precisely when every element is identical.
    """
    return a.size == 0 or float(np.ptp(a)) == 0.0


def _spearman(y_true, y_pred, _t=None) -> float:
    # A constant prediction vector has no defined correlation. Returning NaN is the
    # honest answer; a collapsed model must not be scored as if it had ranked anything.
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
    # average_precision_score, NOT auc(recall, precision): the latter trapezoid-
    # interpolates across the PR curve, which is optimistically biased.
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
    """Coerce inputs to float arrays and reject shape or finiteness problems loudly."""
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
    """Point-estimate regression metrics.

    Spearman matters most for the drug-discovery framing: the practical question is
    whether the model ranks compounds correctly, not whether it nails absolute potency.
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


# ---------------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------------


def _bootstrap_distribution(
    metric: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float | None,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Metric value on each resample; NaN where that resample was degenerate."""
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

    Resamples in which the binarised labels collapse to a single class leave AUC-type
    metrics undefined; those are skipped and counted rather than silently imputed.
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
    """Paired bootstrap on ``metric_a - metric_b``, the headline comparison statistic.

    Both models are scored on the same cached resample matrix, so each resample yields
    a matched pair and the difference is a paired statistic. This is the correct test:
    asking whether two marginal CIs overlap is not, and understates significance.
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
    # Two-sided bootstrap p-value: twice the smaller tail mass either side of zero.
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


# ---------------------------------------------------------------------------------
# Reference predictors.
#
# These return prediction VECTORS, not metrics, so trivial baselines are scored by
# exactly the same code path as the real models. Emitting them unconditionally is what
# stops a mediocre model from looking impressive in isolation.
# ---------------------------------------------------------------------------------


def mean_predictor(y_train: Sequence[float], n_test: int) -> np.ndarray:
    """Constant prediction at the TRAIN mean.

    Deliberately the train mean, not the test mean, so its R^2 is approximately zero
    and may be slightly negative. That is the honest version: a model deployed on
    unseen data does not get to centre itself on that data. Its RMSE is the number
    that makes every other model's R^2 interpretable.
    """
    return np.full(int(n_test), float(np.mean(np.asarray(y_train, dtype=np.float64))))


def majority_class_predictor(
    y_train: Sequence[float], n_test: int, threshold: float = THRESHOLD_PRIMARY
) -> np.ndarray:
    """Constant score placing every test compound in the train-majority class.

    Scored through the same thresholding path as a real model, this yields balanced
    accuracy 0.50, MCC 0.0, and PR-AUC equal to the test positive rate - which is why
    ``pos_rate`` is printed in the same table row.
    """
    yt = np.asarray(y_train, dtype=np.float64)
    majority_is_positive = bool((yt >= threshold).mean() >= 0.5)
    fill = threshold + 1.0 if majority_is_positive else threshold - 1.0
    return np.full(int(n_test), fill)


def random_predictor(
    n_test: int, low: float, high: float, seed: int = BOOTSTRAP_SEED
) -> np.ndarray:
    """Uniform random scores over the observed label range; ROC-AUC should land near 0.50."""
    return np.random.default_rng(seed).uniform(low, high, size=int(n_test))


def one_nn_tanimoto_predict(
    train_fps: Sequence,
    train_y: Sequence[float],
    test_fps: Sequence,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict each test compound's pIC50 from its most ECFP4-similar training compound.

    This is the reference baseline that actually matters: a zero-learning similarity
    lookup. If neither the random forest nor the GNN clearly beats it, the honest
    conclusion is that this dataset is memorisable by similarity and neither model has
    learned structure-activity relationships.

    ``train_fps`` and ``test_fps`` must be RDKit ``ExplicitBitVect`` objects. Returns
    ``(predictions, max_similarity)``; the similarity vector is reused by the leakage
    gates and by the similarity-stratified analysis, so it is returned rather than
    recomputed there.
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


# ---------------------------------------------------------------------------------
# Provenance and the computed verdict
# ---------------------------------------------------------------------------------


def module_sha256() -> str:
    """SHA-256 of this file, stamped into every metrics JSON.

    A gate asserts both models' metrics files carry the same value, which proves they
    were scored by identical code rather than by two copies that quietly drifted.
    """
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def verdict(delta: DeltaResult) -> str:
    """Render the comparison verdict, computed from the CI rather than asserted.

    The README's headline paragraph is generated from this, so the prose cannot drift
    away from the numbers it describes.
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
    """Re-render the verdict straight from a written comparison.json, for the README gate."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return verdict(DeltaResult(**payload["deltas"][metric]))
