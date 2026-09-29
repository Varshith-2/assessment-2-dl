"""Leakage-safe preprocessing: fit on training rows only, then transform.

Mirrors the paper's choices (dummy coding of categoricals, keeping ALL
levels; ordinal survey scales treated as numeric) and adds one documented
assumption: numeric features are standardised, which the paper's
directly-comparable coefficient table implies but never states.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# object-dtype nominal columns in the IBM HR dataset. `Over18` is also
# object-dtype (constant value "Y"); it falls out of the dtype check below
# into the categorical group, becomes a single dummy, and is then removed by
# feature selection -- matching how the paper handles it.
CATEGORICAL_COLS = [
    "BusinessTravel", "Department", "EducationField", "Gender",
    "JobRole", "MaritalStatus", "OverTime", "Over18",
]


def split_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Numeric = numeric dtype; categorical = everything else (object)."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Unfitted transformer. Caller fits it on the training fold only.

    OneHotEncoder(drop=None) keeps every category level, exactly as the
    paper's dummy coding does (e.g. both Gender_Male and Gender_Female);
    the redundancy this creates is what feature selection is meant to
    remove. Constant columns (StandardHours, EmployeeCount, Over18) are
    still present here and are handled downstream by feature selection,
    matching the paper.
    """
    numeric, categorical = split_column_types(X)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(drop=None, handle_unknown="ignore",
                                  sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_transform_frame(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    arr = pre.fit_transform(X)
    return pd.DataFrame(arr, columns=pre.get_feature_names_out(), index=X.index)


def transform_frame(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    arr = pre.transform(X)
    return pd.DataFrame(arr, columns=pre.get_feature_names_out(), index=X.index)
