"""Task 1 -- Exploratory Data Analysis.

Each function answers one question that bears on a preprocessing or
modelling decision in Task 2; nothing is computed "for completeness".
Functions return tidy DataFrames (or a Series) and write a CSV to
outputs/tables. Figures are built by `plotting.py` and assembled in
`run_all`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import ALL_CONSTANT_COLS, ID_COL, ORDINAL_NUMERIC_COLS, TABLE_DIR, TARGET
from data import describe_dataset, load_raw
from plotting import (
    plot_categorical_association, plot_class_balance, plot_correlation_heatmap,
    plot_numeric_association, plot_outliers,
)

# Categorical columns for the feature->target view: every object column
# except the target and the constant `Over18`.
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField", "Gender", "JobRole",
    "MaritalStatus", "OverTime",
]
# Right-skewed monetary / tenure columns for the outlier scan.
OUTLIER_COLS = [
    "MonthlyIncome", "MonthlyRate", "DailyRate", "TotalWorkingYears",
    "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion",
    "YearsWithCurrManager", "NumCompaniesWorked", "DistanceFromHome",
]


def _numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    drop = set(ALL_CONSTANT_COLS) | {ID_COL}
    return [c for c in df.select_dtypes("number").columns if c not in drop]


def _write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(TABLE_DIR / name, index=False)
    return df


# ----------------------------------------------------------------------
# 1. Verification against the paper
# ----------------------------------------------------------------------
def verify_against_paper(df: pd.DataFrame) -> pd.DataFrame:
    d = describe_dataset(df)
    rows = [
        ("Observations", d["n_rows"], "1470", "match"),
        ("Columns (incl. target)", d["n_cols"], "35", "match"),
        ("Feature count", d["n_features_excl_target"], "34 listed",
         "paper drops 2 constants -> 32, and counts EmployeeNumber as a "
         "feature; we treat it as an ID -> 31"),
        ("Constant columns", ", ".join(d["constant_cols"]),
         "StandardHours, EmployeeCount",
         "Over18 is also constant; the paper never names it"),
        ("Missing cells", d["n_missing_cells"], "not discussed",
         "none -> no imputation step"),
        ("Exact duplicate rows", d["duplicate_rows_full"], "not discussed",
         "none"),
        ("Duplicates ignoring EmployeeNumber", d["duplicate_rows_ignoring_id"],
         "not discussed", "none -> rows are distinct employees"),
        ("Target", TARGET, "Attrition", "match"),
        ("Class distribution", str(d["target_counts"]), "imbalanced",
         f"{d['target_positive_rate']:.1%} positive (Yes)"),
    ]
    return _write(pd.DataFrame(rows, columns=["property", "dataset", "paper",
                                              "note"]),
                  "eda_verification.csv")


# ----------------------------------------------------------------------
# 2. Class imbalance
# ----------------------------------------------------------------------
def class_counts(df: pd.DataFrame) -> pd.Series:
    counts = df[TARGET].value_counts()
    _write(counts.rename_axis("class").reset_index(name="count"),
           "eda_class_counts.csv")
    return counts


# ----------------------------------------------------------------------
# 3. Feature -> target association
# ----------------------------------------------------------------------
def numeric_target_association(df: pd.DataFrame) -> pd.DataFrame:
    leave = df[TARGET].eq("Yes")
    rows = []
    for c in _numeric_feature_cols(df):
        a, b = df.loc[leave, c], df.loc[~leave, c]
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({
            "feature": c,
            "mean_stay": round(b.mean(), 2),
            "mean_leave": round(a.mean(), 2),
            "std_mean_diff": round((a.mean() - b.mean()) / pooled, 3)
            if pooled else np.nan,
        })
    out = (pd.DataFrame(rows)
           .sort_values("std_mean_diff", key=np.abs, ascending=False)
           .reset_index(drop=True))
    return _write(out, "eda_numeric_association.csv")


def categorical_target_association(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in CATEGORICAL_COLS:
        rate = df.groupby(c)[TARGET].apply(lambda s: s.eq("Yes").mean())
        rows.append({"feature": c, "levels": len(rate),
                     "min_rate": round(rate.min(), 3),
                     "max_rate": round(rate.max(), 3),
                     "spread": round(rate.max() - rate.min(), 3)})
    out = (pd.DataFrame(rows).sort_values("spread", ascending=False)
           .reset_index(drop=True))
    return _write(out, "eda_categorical_association.csv")


# ----------------------------------------------------------------------
# 4. Redundancy: correlation + VIF
# ----------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[_numeric_feature_cols(df)].corr()


def high_correlations(corr: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    pairs = (corr.where(~np.eye(len(corr), dtype=bool)).stack()
             .rename("r").reset_index())
    pairs = pairs[pairs["level_0"] < pairs["level_1"]]
    out = (pairs[pairs["r"].abs() >= threshold]
           .reindex(pairs["r"].abs().sort_values(ascending=False).index)
           .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
           .reset_index(drop=True))
    return _write(out, "eda_high_correlations.csv")


def vif_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = _numeric_feature_cols(df)
    z = ((df[cols] - df[cols].mean()) / df[cols].std(ddof=0)).dropna(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.DataFrame({
            "feature": z.columns,
            "VIF": [round(variance_inflation_factor(z.values, i), 3)
                    for i in range(z.shape[1])],
        }).sort_values("VIF", ascending=False).reset_index(drop=True)
    return _write(out, "eda_vif.csv")


# ----------------------------------------------------------------------
# 5. Ordinal survey scales treated as numeric (inherited assumption)
# ----------------------------------------------------------------------
def ordinal_scale_check(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in ORDINAL_NUMERIC_COLS:
        vc = df[c].value_counts().sort_index()
        rows.append({"feature": c, "distinct_values": len(vc),
                     "range": f"{df[c].min()}-{df[c].max()}",
                     "value_counts": str(vc.to_dict())})
    return _write(pd.DataFrame(rows), "eda_ordinal_scales.csv")


# ----------------------------------------------------------------------
# 6. Outliers (1.5x IQR) on skewed numeric columns
# ----------------------------------------------------------------------
def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in OUTLIER_COLS:
        q1, q3 = df[c].quantile([0.25, 0.75])
        hi = q3 + 1.5 * (q3 - q1)
        lo = q1 - 1.5 * (q3 - q1)
        mask = (df[c] < lo) | (df[c] > hi)
        rows.append({"feature": c, "n_outliers": int(mask.sum()),
                     "pct": round(100 * mask.mean(), 1),
                     "max": df[c].max(), "upper_fence": round(hi, 1),
                     "skew": round(df[c].skew(), 2)})
    out = pd.DataFrame(rows).sort_values("n_outliers", ascending=False)
    return _write(out, "eda_outliers.csv")


# ----------------------------------------------------------------------
# 7. Leakage audit
# ----------------------------------------------------------------------
def leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        (ID_COL, "row identifier",
         "no predictive meaning; dropped. Paper keeps it as a feature."),
        ("StandardHours / EmployeeCount", "constant",
         "zero information; paper drops before selection, we match."),
        ("Over18", "constant ('Y')",
         "zero information; paper keeps it until selection removes the dummy."),
        ("all remaining features", "employee attributes / history",
         "measured at or before the observation point -- no post-outcome "
         "feature, so no target leakage inside the feature set."),
        ("temporal structure", "none",
         "cross-sectional snapshot, no timestamp -> the brief's time-series "
         "checks (trend, seasonality, autocorrelation) do not apply. The "
         "real leakage risk here is procedural: preprocessing fit before "
         "the split."),
    ]
    return _write(pd.DataFrame(rows, columns=["item", "type", "assessment"]),
                  "eda_leakage_audit.csv")


def run_all(df: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    df = load_raw() if df is None else df
    base_rate = df[TARGET].eq("Yes").mean()
    corr = correlation_matrix(df)
    tables = {
        "verification": verify_against_paper(df),
        "class_counts": class_counts(df).rename_axis("class")
        .reset_index(name="count"),
        "numeric_association": numeric_target_association(df),
        "categorical_association": categorical_target_association(df),
        "high_correlations": high_correlations(corr),
        "vif": vif_table(df),
        "ordinal_scales": ordinal_scale_check(df),
        "outliers": outlier_table(df),
        "leakage": leakage_audit(df),
    }

    plot_class_balance(df[TARGET].value_counts())
    plot_numeric_association(tables["numeric_association"])
    plot_categorical_association(df, CATEGORICAL_COLS, TARGET, base_rate)
    plot_correlation_heatmap(corr)
    plot_outliers(df, OUTLIER_COLS, TARGET)

    print("EDA: 9 tables -> outputs/tables/eda_*.csv, "
          "5 figures -> outputs/figures/0[1-5]_*.png")
    return tables


if __name__ == "__main__":
    t = run_all()
    pd.set_option("display.width", 160)
    print("\n--- verification vs paper ---")
    print(t["verification"].to_string(index=False))
    print("\n--- top VIF ---")
    print(t["vif"].head(10).to_string(index=False))
    print("\n--- outliers ---")
    print(t["outliers"].to_string(index=False))
