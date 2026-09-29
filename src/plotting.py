"""Figures.

Generic - every function takes arrays or frames and hands back a Matplotlib figure, so
nothing in here knows about HDAC1.

Two things are enforced here rather than left to the caller, because both are ways a
comparison quietly flatters itself:

* Parity panels share axis limits and one colour scale. Letting each panel autoscale
  makes a worse model look comparable to a better one.
* Error bars are always the bootstrap interval src/evaluate.py actually computed, never
  a standard error I worked out separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # scripts run headless; the notebook only loads the saved PNGs
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "FIG_DPI",
    "resolve_colour",
    "MODEL_COLOURS",
    "save_figure",
    "parity_panels",
    "learning_curve",
    "stratified_bars",
    "error_distributions",
    "split_diagnostics",
    "importance_split",
]

FIG_DPI = 160
SAVE_KWARGS = {"dpi": FIG_DPI, "bbox_inches": "tight"}

# One colour per model everywhere, so you don't have to re-learn the legend per figure.
MODEL_COLOURS: dict[str, str] = {
    "rf": "#1f6f8b",
    "gnn": "#c1542d",
    "chemprop": "#6a4c93",
    "1nn": "#7a7a7a",
    "mean": "#b9b9b9",
}
SIMILARITY_CMAP = "viridis"


@dataclass(frozen=True)
class Series:
    """One model's predictions on the test fold, ready to plot."""

    label: str
    y_pred: np.ndarray
    colour: str

    def to_dict(self) -> dict:
        return {"label": self.label, "colour": self.colour, "n": int(self.y_pred.size)}


def save_figure(fig: plt.Figure, path: str | Path) -> Path:
    """Write a figure at the project's fixed dpi/bbox settings and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **SAVE_KWARGS)
    plt.close(fig)
    return path


def resolve_colour(label: str) -> str | None:
    """Model label to its fixed colour.

    Substring match on the whole lowercased label, not the first word. First words here
    are "random", "gine" and "chemprop", none of which were colour keys, so a first-word
    lookup quietly returned None and let Matplotlib pick - which is how I ended up with
    two models the same shade of blue.
    """
    key = label.lower()
    for alias, colour in (
        ("random forest", MODEL_COLOURS["rf"]), ("forest", MODEL_COLOURS["rf"]),
        ("rf", MODEL_COLOURS["rf"]),
        ("chemprop", MODEL_COLOURS["chemprop"]), ("d-mpnn", MODEL_COLOURS["chemprop"]),
        ("mpnn", MODEL_COLOURS["chemprop"]),
        ("gine", MODEL_COLOURS["gnn"]), ("gnn", MODEL_COLOURS["gnn"]),
        ("graph", MODEL_COLOURS["gnn"]),
        ("1-nn", MODEL_COLOURS["1nn"]), ("1nn", MODEL_COLOURS["1nn"]),
        ("mean", MODEL_COLOURS["mean"]),
    ):
        if alias in key:
            return colour
    return None


def _shared_limits(*arrays: np.ndarray, pad: float = 0.25) -> tuple[float, float]:
    """Common axis limits across every array, so panels cannot be scaled differently."""
    lo = min(float(np.min(a)) for a in arrays)
    hi = max(float(np.max(a)) for a in arrays)
    return lo - pad, hi + pad


def parity_panels(
    y_true: Sequence[float],
    series: Sequence[Series],
    colour_by: Sequence[float] | None = None,
    colour_label: str = "max Tanimoto to train",
    title: str | None = None,
) -> plt.Figure:
    """Predicted vs true, same axes and colour scale across panels.

    Different axes per panel is the usual way a parity plot lies, so the limits get
    computed once over everything and applied to all of them.
    """
    yt = np.asarray(y_true, dtype=float)
    lim = _shared_limits(yt, *[s.y_pred for s in series])
    norm = None
    if colour_by is not None:
        cb = np.asarray(colour_by, dtype=float)
        norm = matplotlib.colors.Normalize(vmin=float(cb.min()), vmax=float(cb.max()))

    fig, axes = plt.subplots(1, len(series), figsize=(5.2 * len(series), 5.0), squeeze=False)
    scatter = None
    for ax, s in zip(axes[0], series):
        if colour_by is None:
            ax.scatter(yt, s.y_pred, s=10, alpha=0.45, color=s.colour, edgecolors="none")
        else:
            scatter = ax.scatter(
                yt, s.y_pred, c=np.asarray(colour_by, dtype=float), s=10, alpha=0.65,
                cmap=SIMILARITY_CMAP, norm=norm, edgecolors="none",
            )
        ax.plot(lim, lim, ls="--", lw=1.0, color="0.35", zorder=0)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("measured pIC50")
        ax.set_ylabel("predicted pIC50")
        ax.set_title(s.label)
        ax.grid(alpha=0.2, lw=0.5)

    if scatter is not None:
        fig.colorbar(scatter, ax=axes[0].tolist(), shrink=0.85, label=colour_label)
    if title:
        fig.suptitle(title, y=1.02)
    return fig


def learning_curve(
    fractions: Sequence[float],
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    n_train_total: int,
    ylabel: str = "test RMSE (pIC50 log units)",
    title: str | None = None,
) -> plt.Figure:
    """Test error vs training-set size, mean and +/- 1 sd band per model.

    curves maps a label to (means, sds) across fractions. The x axis shows compounds as
    well as percent - "is this data-limited?" is a question about actual n, not a ratio.
    """
    fr = np.asarray(fractions, dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for label, (means, sds) in curves.items():
        m = np.asarray(means, dtype=float)
        s = np.asarray(sds, dtype=float)
        # Draw the line first and read its actual colour back, so the sd band can't end up
        # a different colour from its own line.
        (line,) = ax.plot(fr * 100, m, marker="o", lw=1.8, label=label, color=resolve_colour(label))
        ax.fill_between(fr * 100, m - s, m + s, alpha=0.18, color=line.get_color(), lw=0)
    ax.set_xlabel("training set used (%)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)

    secondary = ax.secondary_xaxis(
        "top", functions=(lambda p: p / 100 * n_train_total, lambda n: n / n_train_total * 100)
    )
    secondary.set_xlabel("training compounds")
    if title:
        ax.set_title(title)
    return fig


def stratified_bars(
    bin_labels: Sequence[str],
    values: Mapping[str, Sequence[float]],
    errors: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    counts: Sequence[int] | None = None,
    ylabel: str = "test RMSE (pIC50 log units)",
    xlabel: str = "max Tanimoto similarity to training set",
    title: str | None = None,
) -> plt.Figure:
    """Grouped bars per similarity bin, with asymmetric bootstrap intervals.

    errors maps a label to (lo, hi) bounds, converted to the offsets Matplotlib wants.
    Passing the real interval instead of a symmetric sd keeps the drawn bar the same as
    the number in the table.
    """
    labels = list(values)
    n_groups = len(bin_labels)
    width = 0.8 / max(len(labels), 1)
    x = np.arange(n_groups, dtype=float)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for i, label in enumerate(labels):
        v = np.asarray(values[label], dtype=float)
        offset = (i - (len(labels) - 1) / 2) * width
        yerr = None
        if errors and label in errors:
            lo = np.asarray(errors[label][0], dtype=float)
            hi = np.asarray(errors[label][1], dtype=float)
            yerr = np.vstack([np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)])
        ax.bar(
            x + offset, v, width=width * 0.92, label=label,
            color=resolve_colour(label),
            yerr=yerr, capsize=3, error_kw={"lw": 1.0, "ecolor": "0.3"},
        )
    ticks = list(bin_labels)
    if counts is not None:
        ticks = [f"{b}\n(n={c})" for b, c in zip(bin_labels, counts)]
    ax.set_xticks(x)
    ax.set_xticklabels(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    return fig


def error_distributions(
    residuals: Mapping[str, Sequence[float]],
    pair: tuple[str, str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Absolute-residual ECDFs, plus a model-vs-model residual scatter.

    The scatter is the useful one. If the residuals correlate strongly, both models are
    losing on the same compounds, which points at the data rather than the architectures.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    for label, res in residuals.items():
        a = np.sort(np.abs(np.asarray(res, dtype=float)))
        ax.plot(
            a, np.arange(1, a.size + 1) / a.size, lw=1.8, label=label,
            color=resolve_colour(label),
        )
    ax.set_xlabel("|residual| (pIC50 log units)")
    ax.set_ylabel("cumulative fraction of test set")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    ax.set_title("Absolute error ECDF")

    ax = axes[1]
    if pair is not None:
        a = np.asarray(residuals[pair[0]], dtype=float)
        b = np.asarray(residuals[pair[1]], dtype=float)
        lim = _shared_limits(a, b)
        ax.scatter(a, b, s=10, alpha=0.4, color="#4c4c4c", edgecolors="none")
        ax.plot(lim, lim, ls="--", lw=1.0, color="0.35")
        ax.axhline(0, lw=0.8, color="0.6")
        ax.axvline(0, lw=0.8, color="0.6")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"{pair[0]} residual")
        ax.set_ylabel(f"{pair[1]} residual")
        r = float(np.corrcoef(a, b)[0, 1])
        ax.set_title(f"Residual agreement (Pearson r = {r:.2f})")
        ax.grid(alpha=0.2, lw=0.5)
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


def split_diagnostics(
    fold_labels: Sequence[str],
    fold_values: Mapping[str, Sequence[float]],
    group_sizes: Sequence[int],
    max_similarity: Sequence[float],
    flag_threshold: float = 0.85,
) -> plt.Figure:
    """The leak-free argument as a picture: label shift, group sizes, NN similarity.

    First figure in the README, because it's the evidence the split did what I say it
    did, and you should be able to check that before reading any model number.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    ax = axes[0]
    for fold in fold_labels:
        ax.hist(
            np.asarray(fold_values[fold], dtype=float), bins=40, density=True,
            histtype="step", lw=1.8, label=f"{fold} (n={len(fold_values[fold])})",
        )
    ax.set_xlabel("pIC50")
    ax.set_ylabel("density")
    ax.set_title("Label distribution by fold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    gs = np.asarray(group_sizes, dtype=int)
    bins = np.logspace(0, np.log10(max(gs.max(), 2)), 30)
    ax.hist(gs, bins=bins, color="#4c78a8")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compounds per Murcko scaffold group")
    ax.set_ylabel("number of groups")
    ax.set_title(f"Scaffold group sizes (n={gs.size} groups)")
    ax.grid(alpha=0.25, lw=0.5)

    ax = axes[2]
    ms = np.asarray(max_similarity, dtype=float)
    ax.hist(ms, bins=40, color="#7a9e5c")
    ax.axvline(
        flag_threshold, ls="--", lw=1.4, color="#c1542d",
        label=f"flag >= {flag_threshold} ({float((ms >= flag_threshold).mean()) * 100:.1f}%)",
    )
    ax.axvline(
        float(np.median(ms)), ls=":", lw=1.4, color="0.25",
        label=f"median {float(np.median(ms)):.2f}",
    )
    ax.set_xlabel("max ECFP4 Tanimoto to any training compound")
    ax.set_ylabel("test compounds")
    ax.set_title("Test-to-train nearest-neighbour similarity")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)

    fig.tight_layout()
    return fig


def importance_split(
    fingerprint_importance: float,
    descriptor_importance: float,
    top_descriptors: Mapping[str, float] | None = None,
) -> plt.Figure:
    """Where the forest's importance goes: fingerprint bits vs the 12 descriptors.

    12 descriptors are competing with 2048 bits for split candidates. If the descriptors
    carry most of it, the headline is really about bulk properties, not substructure.
    """
    ncols = 1 if not top_descriptors else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 4.4), squeeze=False)

    ax = axes[0][0]
    ax.bar(
        ["ECFP4 bits\n(2048 features)", "descriptors\n(12 features)"],
        [fingerprint_importance, descriptor_importance],
        color=["#4c78a8", "#e4a33c"],
    )
    ax.set_ylabel("summed Gini importance")
    ax.set_title("Where the forest looks")
    ax.grid(alpha=0.25, lw=0.5, axis="y")

    if top_descriptors:
        ax = axes[0][1]
        names = list(top_descriptors)[::-1]
        vals = [top_descriptors[n] for n in names]
        ax.barh(names, vals, color="#e4a33c")
        ax.set_xlabel("Gini importance")
        ax.set_title("Descriptor importances")
        ax.grid(alpha=0.25, lw=0.5, axis="x")

    fig.tight_layout()
    return fig


def bin_by_similarity(
    max_similarity: Sequence[float],
    edges: Iterable[float] = (0.0, 0.3, 0.4, 0.5, 0.7, 1.0001),
) -> tuple[np.ndarray, list[str]]:
    """Assign each test compound to a similarity bin; returns indices and pretty labels."""
    e = np.asarray(list(edges), dtype=float)
    idx = np.digitize(np.asarray(max_similarity, dtype=float), e[1:-1], right=False)
    labels = [f"{e[i]:.2f}-{min(e[i + 1], 1.0):.2f}" for i in range(len(e) - 1)]
    return idx, labels
