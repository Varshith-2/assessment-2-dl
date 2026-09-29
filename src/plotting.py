"""Shared visual identity and figure builders for the whole project.

Everything that draws lives here so the EDA and the replication share one
palette, one theme, and one `save_fig`. Analysis code in `eda.py` /
`experiment.py` returns tidy DataFrames; this module turns them into
figures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from config import FIGURE_DIR

# --------------------------------------------------------------------------
# Palette -- one source of truth. Semantic names, not hex codes, at call
# sites.
# --------------------------------------------------------------------------
PALETTE = {
    "stay":    "#3B6EA5",   # negative class / "higher among stayers"
    "leave":   "#C1443C",   # positive class / "higher among leavers"
    "lr":      "#3B6EA5",   # Logistic Regression
    "rf":      "#C1443C",   # Random Forest
    "neutral": "#6B7280",   # reference lines, de-emphasised marks
    "accent":  "#E1A948",   # highlight (selected threshold, callouts)
    "grid":    "#D8DCE1",
}
SEQ_CMAP = "RdBu_r"          # diverging, for correlation matrices
INK = "#20242B"


def apply_theme() -> None:
    """Project-wide matplotlib defaults. Call once at import."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["neutral"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": PALETTE["neutral"],
        "ytick.color": PALETTE["neutral"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
    })


apply_theme()


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Write to outputs/figures and close. Returns the path."""
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _label_barh(ax, values, *, fmt="{:.0%}", pad=0.01, color=INK):
    """Value labels at the end of horizontal bars."""
    span = ax.get_xlim()[1] - ax.get_xlim()[0]
    for y, v in enumerate(values):
        ax.text(v + pad * span * np.sign(v or 1), y, fmt.format(v),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=8, color=color)


# ======================================================================
# EDA figures
# ======================================================================
def plot_class_balance(counts: pd.Series, name="01_class_balance.png") -> Path:
    """Single 100%-stacked bar: the imbalance at a glance."""
    total = counts.sum()
    stay, leave = counts["No"], counts["Yes"]
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    ax.barh([0], [stay / total], color=PALETTE["stay"], label="stays (No)")
    ax.barh([0], [leave / total], left=[stay / total], color=PALETTE["leave"],
            label="leaves (Yes)")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.text(stay / total / 2, 0, f"{stay:,}\n{stay / total:.1%}", ha="center",
            va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(stay / total + leave / total / 2, 0,
            f"{leave:,}\n{leave / total:.1%}", ha="center", va="center",
            color="white", fontweight="bold", fontsize=10)
    ax.set_title("Attrition is a 1-in-6 minority class "
                 "(accuracy alone would be misleading)")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.55))
    for s in ("bottom", "left"):
        ax.spines[s].set_visible(False)
    return save_fig(fig, name)


def plot_numeric_association(assoc: pd.DataFrame,
                             name="02_numeric_target_association.png") -> Path:
    """Standardised mean difference (Cohen's d) per numeric feature."""
    d = assoc.iloc[::-1]
    colors = [PALETTE["leave"] if v > 0 else PALETTE["stay"]
              for v in d["std_mean_diff"]]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh(d["feature"], d["std_mean_diff"], color=colors)
    ax.axvline(0, color=INK, lw=1)
    lim = float(np.abs(assoc["std_mean_diff"]).max()) * 1.25
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Cohen's d  (mean$_{leave}$ - mean$_{stay}$, pooled SD)")
    ax.set_title("Which numeric features separate leavers from stayers")
    ax.text(-lim * 0.97, len(d) - 0.5, "higher among stayers",
            fontsize=8, color=PALETTE["stay"], style="italic")
    ax.text(lim * 0.97, len(d) - 0.5, "higher among leavers", ha="right",
            fontsize=8, color=PALETTE["leave"], style="italic")
    ax.grid(axis="x", alpha=0.6)
    return save_fig(fig, name)


def plot_categorical_association(df: pd.DataFrame, cat_cols, target, base_rate,
                                 name="03_categorical_target_association.png"
                                 ) -> Path:
    """One panel per categorical feature, attrition rate by level.

    All panels share the x-axis so bar lengths are comparable across
    panels; the grid fits every feature (previous version silently
    dropped the 7th)."""
    n = len(cat_cols)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    x_max = 0.0
    rates = {}
    for c in cat_cols:
        r = df.groupby(c)[target].apply(lambda s: s.eq("Yes").mean()).sort_values()
        rates[c] = r
        x_max = max(x_max, r.max())
    x_max = min(1.0, x_max * 1.15)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.5 * nrows),
                             squeeze=False)
    for ax, c in zip(axes.flat, cat_cols):
        r = rates[c]
        colors = [PALETTE["leave"] if v >= base_rate else PALETTE["stay"]
                  for v in r]
        ax.barh(range(len(r)), r.values, color=colors)
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels([str(i)[:26] for i in r.index], fontsize=8)
        ax.axvline(base_rate, color=INK, ls="--", lw=1)
        ax.set_xlim(0, x_max)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.set_title(c, fontsize=10, loc="left")
        for y, v in enumerate(r.values):
            # Label sits inside the bar when its end is near the baseline
            # line, otherwise just past the bar end -- avoids collisions.
            near_base = abs(v - base_rate) < 0.02 * x_max + 0.012
            ax.text(v + (-0.006 if near_base else 0.006), y, f"{v:.0%}",
                    va="center", ha="right" if near_base else "left",
                    fontsize=7.5, color="white" if near_base else INK)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"Attrition rate by category   "
                 f"(dashed line = overall {base_rate:.1%})",
                 fontsize=13, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return save_fig(fig, name)


def plot_correlation_heatmap(corr: pd.DataFrame,
                             name="04_correlation_heatmap.png") -> Path:
    """Lower-triangle heatmap, features ordered by hierarchical clustering
    so correlated blocks sit together; |r| >= 0.5 annotated."""
    dist = 1 - corr.abs()
    order = leaves_list(linkage(squareform(dist.values, checks=False),
                                method="average"))
    c = corr.iloc[order, order]
    mask = np.triu(np.ones_like(c, dtype=bool), k=1)
    cm = c.mask(mask)

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm.values, cmap=SEQ_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels(c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels(c.index, fontsize=8)
    for i in range(len(c)):
        for j in range(i + 1):
            v = c.values[i, j]
            if i != j and abs(v) >= 0.5:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Pearson r", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("Numeric-feature correlation  "
                 "(clustered order; |r| >= 0.5 labelled)")
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(-.5, len(c)), minor=True)
    ax.set_yticks(np.arange(-.5, len(c)), minor=True)
    ax.grid(which="minor", color="white", lw=1)
    ax.tick_params(which="minor", length=0)
    return save_fig(fig, name)


def plot_outliers(df: pd.DataFrame, cols, target,
                  name="05_outliers_strip.png") -> Path:
    """Per-feature z-score strips, split by attrition, with the 1.5x IQR
    fence marked. Answers 'are the long right tails linked to leaving?'."""
    z = df[cols].apply(lambda s: (s - s.mean()) / s.std())
    leave = df[target].eq("Yes").values
    fig, ax = plt.subplots(figsize=(9.5, 5))
    rng = np.random.RandomState(0)
    for i, c in enumerate(cols):
        for grp, colr, off in [(~leave, PALETTE["stay"], -0.16),
                               (leave, PALETTE["leave"], 0.16)]:
            vals = z[c].values[grp]
            ax.scatter(vals, np.full(vals.sum() if vals.dtype == bool
                                     else len(vals), i + off)
                       + rng.uniform(-0.09, 0.09, len(vals)),
                       s=8, alpha=0.35, color=colr, edgecolors="none")
        q1, q3 = np.percentile(z[c], [25, 75])
        ax.plot([q3 + 1.5 * (q3 - q1)] * 2, [i - 0.3, i + 0.3],
                color=INK, lw=1.2)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.invert_yaxis()
    ax.set_xlabel("z-score (per feature)")
    ax.set_title("Right-tail outliers in skewed features, split by attrition\n"
                 "vertical tick = upper 1.5x IQR fence")
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE["stay"],
                          label="stays"),
               plt.Line2D([], [], marker="o", ls="", color=PALETTE["leave"],
                          label="leaves")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.6)
    return save_fig(fig, name)


# ======================================================================
# Replication figures
# ======================================================================
def plot_threshold_and_pr(curves: dict, base_rate: float,
                          name="06_fair_model_comparison.png") -> Path:
    """(a) test F1 vs decision threshold, (b) precision-recall curves.

    `curves[model] = dict(grid, f1, thr, precision, recall, pr_auc,
                          op_precision, op_recall)`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    key = {"LogisticRegression": "lr", "RandomForest": "rf"}
    for model, d in curves.items():
        col = PALETTE[key[model]]
        axes[0].plot(d["grid"], d["f1"], color=col, lw=2,
                     label=f"{model} + max-out")
        axes[0].axvline(d["thr"], color=col, ls=":", lw=1.4)
        axes[0].scatter([d["thr"]], [np.interp(d["thr"], d["grid"], d["f1"])],
                        color=col, zorder=5, s=35)
        axes[1].plot(d["recall"], d["precision"], color=col, lw=2,
                     label=f"{model}  (PR-AUC {d['pr_auc']:.3f})")
        axes[1].scatter([d["op_recall"]], [d["op_precision"]], color=col,
                        zorder=5, s=45, edgecolors="white")

    axes[0].axvline(0.5, color=PALETTE["neutral"], ls="--", lw=1,
                    label="fixed 0.5 cut (paper)")
    axes[0].set(xlabel="decision threshold", ylabel="test F1",
                title="(a) F1 vs threshold  "
                      "(dotted = threshold picked on validation)")
    axes[0].legend(loc="lower center")
    axes[0].grid(alpha=0.6)

    axes[1].axhline(base_rate, color=PALETTE["neutral"], ls="--", lw=1,
                    label=f"no-skill ({base_rate:.3f})")
    axes[1].set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1),
                title="(b) Precision-recall on the test set\n"
                      "dot = chosen operating point")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.6)
    fig.tight_layout()
    return save_fig(fig, name)


def plot_bootstrap_coefficients(boot: pd.DataFrame, *, top: int = 20,
                                name="07_bootstrap_coefficients.png") -> Path:
    """Forest plot of the LR + max-out coefficients over 300 bootstraps:
    mean +/- 1 SD, ordered by magnitude, coloured by stability (CV)."""
    d = (boot.assign(absmean=boot["mean"].abs())
         .sort_values("absmean", ascending=False).head(top)
         .sort_values("mean"))
    cv = d["coef_of_variation"].abs()
    stable = cv < 0.5
    colors = np.where(stable, PALETTE["stay"], PALETTE["accent"])

    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(d) + 1.2))
    ax.errorbar(d["mean"], range(len(d)), xerr=d["std"], fmt="o", ms=5,
                lw=0, elinewidth=1.5, ecolor=PALETTE["neutral"],
                mfc="none", mec=PALETTE["neutral"])
    ax.scatter(d["mean"], range(len(d)), c=colors, s=42, zorder=5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["feature"], fontsize=8)
    ax.set_xlabel("logistic-regression coefficient (standardised features)")
    ax.set_title("Coefficient stability over 300 bootstrap refits\n"
                 "mean +/- 1 SD; amber = coefficient of variation >= 0.5")
    handles = [plt.Line2D([], [], marker="o", ls="", color=PALETTE["stay"],
                          label="stable (CV < 0.5)"),
               plt.Line2D([], [], marker="o", ls="", color=PALETTE["accent"],
                          label="unstable (CV >= 0.5)")]
    ax.legend(handles=handles, loc="lower right")
    ax.grid(axis="x", alpha=0.6)
    return save_fig(fig, name)
