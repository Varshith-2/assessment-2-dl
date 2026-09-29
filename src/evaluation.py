"""Metric helpers shared by the replication experiment: point metrics,
validation-tuned decision threshold, and the two bootstrap procedures
(test-metric CIs, and the paper's coefficient-confidence analysis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)

from config import N_BOOTSTRAP, N_TEST_BOOTSTRAP, RANDOM_STATE


def point_metrics(y_true, y_pred) -> dict:
    """Accuracy / precision / recall / F1 at a fixed set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def ranking_metrics(y_true, y_score) -> dict:
    """Threshold-free metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def best_threshold(y_val, proba_val) -> tuple[float, float]:
    """Probability cut that maximises F1 on the validation set. Returns
    (threshold, validation_F1). Applied identically to every model so the
    comparison is at a comparable operating point rather than a fixed 0.5."""
    grid = np.unique(np.round(proba_val, 4))
    grid = grid[(grid > 0.01) & (grid < 0.99)]
    if grid.size == 0:
        return 0.5, float("nan")
    scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0)
              for t in grid]
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def bootstrap_metric_ci(y_true, y_pred, *, metric=f1_score, n=N_TEST_BOOTSTRAP,
                        seed=RANDOM_STATE, pct=(2.5, 97.5)) -> tuple[float, float]:
    """Percentile bootstrap CI for a metric, resampling (y_true, y_pred)
    pairs. `y_pred` is already thresholded, so this works for any frozen
    prediction vector."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.RandomState(seed)
    n_obs = len(y_true)
    stats = np.empty(n)
    for b in range(n):
        idx = rng.randint(0, n_obs, n_obs)
        stats[b] = metric(y_true[idx], y_pred[idx], zero_division=0)
    lo, hi = np.percentile(stats, pct)
    return round(float(lo), 4), round(float(hi), 4)


def bootstrap_coefficients(estimator, X, y, feature_names, *, n=N_BOOTSTRAP,
                           seed=RANDOM_STATE) -> pd.DataFrame:
    """The paper's Section 5.3 analysis: refit `estimator` on `n` bootstrap
    resamples of the training data and summarise each coefficient by mean,
    SD and coefficient of variation (SD / |mean|). Low CV => stable.

    `estimator` is cloned each fit, so the exact model configuration used
    for the headline result is the one bootstrapped here."""
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    n_obs = len(y)
    coefs = np.empty((n, X.shape[1]))
    for b in range(n):
        idx = rng.randint(0, n_obs, n_obs)
        model = clone(estimator).fit(X[idx], y[idx])
        coefs[b] = model.coef_[0]
    mean = coefs.mean(axis=0)
    sd = coefs.std(axis=0, ddof=1)
    cv = np.where(np.abs(mean) > 1e-9, sd / np.abs(mean), np.nan)
    return (pd.DataFrame({"feature": feature_names, "mean": mean, "std": sd,
                          "coef_of_variation": cv})
            .sort_values("coef_of_variation", key=np.abs)
            .reset_index(drop=True))
