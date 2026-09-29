"""Figures 08-12: evidence for the Assignment #3 hypotheses (Step 3).

Each figure exists to make one EDA -> proposed-change link visible, not to
decorate. They reuse the shared theme/palette from `plotting.py` and are
referenced from `report/hypotheses.md`. Run after `eda.py` (needs nothing
that `eda.py` doesn't already load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from config import ORDINAL_NUMERIC_COLS, TARGET
from data import load_raw
from eda import CATEGORICAL_COLS, _numeric_feature_cols, correlation_matrix
from plotting import INK, PALETTE, SEQ_CMAP, save_fig

import matplotlib.pyplot as plt
import matplotlib as mpl

SKEWED_COLS = ["MonthlyIncome", "YearsSinceLastPromotion", "YearsAtCompany"]
VIF_BLOCK = ["JobLevel", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
             "YearsInCurrentRole", "YearsWithCurrManager", "PercentSalaryHike",
             "PerformanceRating", "Age", "YearsSinceLastPromotion"]


# ---------------------------------------------------------------- H1: skew
def plot_skew_transform(df: pd.DataFrame,
                        name="08_skew_transform.png"):
    """Raw vs log1p histograms for the three most skewed columns -- does a
    transform actually pull in the tail?"""
    fig, axes = plt.subplots(2, len(SKEWED_COLS), figsize=(11, 5.5))
    for j, c in enumerate(SKEWED_COLS):
        raw = df[c].astype(float)
        logged = np.log1p(raw)
        for i, (vals, label) in enumerate([(raw, "raw"), (logged, "log1p")]):
            ax = axes[i, j]
            ax.hist(vals, bins=30, color=PALETTE["lr"] if i == 0 else PALETTE["accent"],
                    alpha=0.85, edgecolor="white", linewidth=0.4)
            sk = stats.skew(vals)
            ax.set_title(f"{c}\n{label}  (skew={sk:.2f})", fontsize=9)
            ax.grid(axis="y", alpha=0.4)
            if j == 0:
                ax.set_ylabel("count")
    fig.suptitle("H1 -- log1p pulls in the right tail of the skewed monetary/"
                 "tenure columns",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_fig(fig, name)


# --------------------------------------------------------- H2: collinearity
def plot_collinearity_focus(df: pd.DataFrame,
                            name="09_collinearity_focus.png"):
    """Zoomed heatmap on exactly the high-VIF block, annotated in full
    (not just |r|>=0.5) so the JobLevel/MonthlyIncome near-duplication and
    the Years* block are both readable at once."""
    corr = correlation_matrix(df)[VIF_BLOCK].loc[VIF_BLOCK]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap=SEQ_CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Pearson r", fontsize=9)
    cb.outline.set_visible(False)
    ax.set_title("H2 -- the high-VIF block: JobLevel~MonthlyIncome (r=0.95) "
                 "and a correlated Years* cluster")
    ax.spines[:].set_visible(False)
    fig.tight_layout()
    return save_fig(fig, name)


# ---------------------------------------------------- H3: ordinal-as-numeric
def plot_ordinal_attrition_rate(df: pd.DataFrame,
                                name="10_ordinal_attrition_rate.png"):
    """Attrition rate at every level of every ordinal scale. A monotonic
    staircase supports treating the scale as numeric; a non-monotonic
    (U-shaped / flat-then-drop) pattern is evidence the equal-spacing
    assumption is wrong and one-hot may serve better."""
    base_rate = df[TARGET].eq("Yes").mean()
    cols = ORDINAL_NUMERIC_COLS
    ncols = 3
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows), squeeze=False)
    for ax, c in zip(axes.flat, cols):
        rate = df.groupby(c)[TARGET].apply(lambda s: s.eq("Yes").mean())
        levels = rate.index.astype(str)
        diffs = np.diff(rate.values)
        monotonic = np.all(diffs >= -1e-9) or np.all(diffs <= 1e-9)
        colors = [PALETTE["stay"] if monotonic else PALETTE["accent"]] * len(rate)
        ax.bar(levels, rate.values, color=colors)
        ax.axhline(base_rate, color=INK, ls="--", lw=1)
        ax.set_title(f"{c}{'  (monotonic)' if monotonic else '  (NOT monotonic)'}",
                    fontsize=8.5)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
        ax.tick_params(labelsize=7.5)
        ax.grid(axis="y", alpha=0.4)
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("H3 -- is attrition rate a straight line across each ordinal "
                "scale? (dashed = overall 16.1%; grey=steady, gold=uneven)",
                fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save_fig(fig, name)


# -------------------------------------------------------------- H5: scaling
def plot_scaling_sensitivity(df: pd.DataFrame,
                             name="11_scaling_outlier_sensitivity.png"):
    """StandardScaler vs a robust (median/IQR) scaling on the same skewed
    column: how far do the outliers get pushed out under each scheme?"""
    fig, axes = plt.subplots(1, len(SKEWED_COLS), figsize=(11, 3.2))
    for ax, c in zip(axes, SKEWED_COLS):
        raw = df[c].astype(float)
        z = (raw - raw.mean()) / raw.std(ddof=0)
        med, iqr = raw.median(), raw.quantile(.75) - raw.quantile(.25)
        rz = (raw - med) / iqr
        ax.scatter(z, np.zeros(len(z)) + 0.15, s=10, alpha=0.35,
                   color=PALETTE["lr"], label="StandardScaler")
        ax.scatter(rz, np.zeros(len(rz)) - 0.15, s=10, alpha=0.35,
                   color=PALETTE["accent"], label="Robust (median/IQR)")
        ax.set_yticks([0.15, -0.15])
        ax.set_yticklabels(["Standard", "Robust"], fontsize=8)
        ax.set_title(f"{c}\nmax |scaled value|: "
                    f"{z.abs().max():.1f} vs {rz.abs().max():.1f}", fontsize=8.5)
        ax.axvline(0, color=INK, lw=0.6)
        ax.grid(axis="x", alpha=0.4)
    fig.suptitle("H5 -- does a robust (median/IQR) scaling change how far "
                "the same outliers sit from the bulk of the data?",
                fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_fig(fig, name)


# ------------------------------------------------------ H6: engineered ratio
def plot_feature_engineering_ratio(df: pd.DataFrame,
                                   name="12_feature_engineering_ratio.png"):
    """TotalWorkingYears vs YearsAtCompany (the strongest, most collinear
    tenure pair) colour-coded by attrition, plus the derived 'company
    tenure share' ratio split by attrition -- does the ratio separate the
    classes at least as well as the two raw columns?"""
    leave = df[TARGET].eq("Yes")
    ratio = (df["YearsAtCompany"] / df["TotalWorkingYears"].replace(0, np.nan)
            ).clip(upper=1).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    ax.scatter(df.loc[~leave, "TotalWorkingYears"], df.loc[~leave, "YearsAtCompany"],
              s=14, alpha=0.35, color=PALETTE["stay"], label="stays")
    ax.scatter(df.loc[leave, "TotalWorkingYears"], df.loc[leave, "YearsAtCompany"],
              s=14, alpha=0.5, color=PALETTE["leave"], label="leaves")
    ax.plot([0, 40], [0, 40], color=INK, lw=0.8, ls="--")
    ax.set_xlabel("TotalWorkingYears")
    ax.set_ylabel("YearsAtCompany")
    ax.set_title("Raw pair (r=0.77, both high-VIF)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(ratio[~leave], bins=bins, color=PALETTE["stay"], alpha=0.6,
           density=True, label="stays")
    ax.hist(ratio[leave], bins=bins, color=PALETTE["leave"], alpha=0.6,
           density=True, label="leaves")
    ax.set_xlabel("YearsAtCompany / TotalWorkingYears  (career share at this employer)")
    ax.set_ylabel("density")
    ax.set_title("Engineered ratio, one column instead of two")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("H6 -- a single tenure-share ratio vs the two raw, "
                "correlated tenure columns",
                fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_fig(fig, name)


def run_all():
    df = load_raw()
    paths = [
        plot_skew_transform(df),
        plot_collinearity_focus(df),
        plot_ordinal_attrition_rate(df),
        plot_scaling_sensitivity(df),
        plot_feature_engineering_ratio(df),
    ]
    for p in paths:
        print("wrote", p)
    return paths


if __name__ == "__main__":
    run_all()
