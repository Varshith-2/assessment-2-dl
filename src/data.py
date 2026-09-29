"""Dataset loading, verification, and the leakage-safe splitter."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH, ID_COL, PAPER_DROPPED_CONSTANTS, RANDOM_STATE, TARGET,
    TEST_SIZE, VAL_SIZE,
)


def load_raw() -> pd.DataFrame:
    """Load the IBM HR CSV. The file carries a UTF-8 BOM on the first header."""
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


def describe_dataset(df: pd.DataFrame) -> dict:
    """Structural facts the assignment asks us to report and to check
    against the paper's description."""
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    dup_full = int(df.duplicated().sum())
    dup_no_id = int(df.drop(columns=[ID_COL]).duplicated().sum())
    y = df[TARGET].map({"No": 0, "Yes": 1})
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "n_features_excl_target": df.shape[1] - 1,
        "dtypes": df.dtypes.astype(str).value_counts().to_dict(),
        "n_missing_cells": int(df.isna().sum().sum()),
        "cols_with_missing": df.columns[df.isna().any()].tolist(),
        "duplicate_rows_full": dup_full,
        "duplicate_rows_ignoring_id": dup_no_id,
        "constant_cols": constant_cols,
        "target_counts": df[TARGET].value_counts().to_dict(),
        "target_positive_rate": float(y.mean()),
    }


def get_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X, y with the target mapped to 0/1, the pure identifier
    column removed, and the two constant columns the paper explicitly drops
    (StandardHours, EmployeeCount) removed. Over18 is intentionally KEPT --
    the paper carries it through to feature selection, which then removes
    its dummy."""
    y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X = df.drop(columns=[TARGET, ID_COL, *PAPER_DROPPED_CONSTANTS])
    return X, y


def three_way_split(X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE):
    """Stratified train / validation / test split done BEFORE any encoding,
    scaling, imputation, or feature selection. This is correction (A) to the
    paper's methodology: nothing downstream may see the test rows.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First peel off the test set.
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=random_state,
    )
    # Then split the remainder into train and validation. VAL_SIZE is a
    # fraction of the FULL dataset, so rescale it to the remainder.
    val_fraction_of_remainder = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_fraction_of_remainder, stratify=y_tv,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_summary(y_train, y_val, y_test) -> pd.DataFrame:
    rows = []
    for name, s in [("train", y_train), ("val", y_val), ("test", y_test)]:
        rows.append({
            "split": name, "n": len(s), "n_pos": int(s.sum()),
            "positive_rate": round(float(s.mean()), 4),
        })
    return pd.DataFrame(rows)
