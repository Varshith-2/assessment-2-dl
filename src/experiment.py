"""Task 2 -- replicate the paper's pipeline with two methodological
corrections, and reproduce its post-processing coefficient analysis.

Correction A -- split first. The stratified train / validation / test split
happens before any encoding, scaling, or feature selection. Preprocessing is
fit on the training rows only; the test set is scored exactly once.

Correction B -- a fair comparison. The paper pits its Logistic Regression +
max-out against classifier numbers lifted from other papers with different
splits and preprocessing. Instead we run Logistic Regression and Random
Forest on one identical split, seed, and preprocessor, each with its own
max-out search, and -- because `class_weight='balanced'` shifts LR's
decision boundary but not RF's majority vote -- we pick each model's
probability threshold on the validation set so both are compared at a
comparable operating point. Threshold-free ROC-AUC / PR-AUC are reported
alongside.

All heavy logic is imported from sibling modules; this file orchestrates.
Run directly (`python experiment.py`) to regenerate every table and figure.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve

from config import CACHE_DIR, RANDOM_STATE, TABLE_DIR
from data import get_feature_frame, load_raw, three_way_split
from evaluation import (
    best_threshold, bootstrap_coefficients, bootstrap_metric_ci,
    point_metrics, ranking_metrics,
)
from maxout import maxout_select
from plotting import plot_bootstrap_coefficients, plot_threshold_and_pr
from preprocessing import build_preprocessor, fit_transform_frame, transform_frame

# numpy here links against Apple's Accelerate BLAS, whose vectorised matmul
# raises spurious overflow / divide-by-zero FPU flags inside the lbfgs
# solve. Verified benign: coefficients stay finite, the solver converges in
# ~70 of 5000 iterations, and lbfgs and liblinear agree on 99.7% of
# predictions. Filtered so genuine warnings stay visible.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

# Paper's own reported test results (Section 5.2), for the comparison table.
PAPER_RESULTS = [
    {"model": "LogReg (paper) + max-out", "variant": "paper",
     "accuracy": 0.81, "precision": 0.43, "recall": 0.82, "f1": 0.56},
    {"model": "LogReg (paper), all features", "variant": "paper",
     "accuracy": 0.78, "precision": 0.39, "recall": 0.82, "f1": 0.53},
]


@dataclass
class PreparedData:
    """One split, one preprocessor, everything downstream needs.

    `*_t` frames are standardised + dummy-coded, with the encoder fit on
    the training rows only. `pool_*` is train+validation, used only as the
    resampling pool for the max-out feature search (never the test set).
    """
    X_train_t: pd.DataFrame
    X_val_t: pd.DataFrame
    X_test_t: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    pool_raw: pd.DataFrame
    pool_y: pd.Series
    feature_names: list[str]


def prepare_data() -> PreparedData:
    df = load_raw()
    X, y = get_feature_frame(df)
    X_train, X_val, X_test, y_train, y_val, y_test = three_way_split(X, y)

    pre = build_preprocessor(X_train)
    X_train_t = fit_transform_frame(pre, X_train)
    return PreparedData(
        X_train_t=X_train_t,
        X_val_t=transform_frame(pre, X_val),
        X_test_t=transform_frame(pre, X_test),
        y_train=y_train, y_val=y_val, y_test=y_test,
        pool_raw=pd.concat([X_train, X_val]),
        pool_y=pd.concat([y_train, y_val]),
        feature_names=list(X_train_t.columns),
    )


def make_models() -> dict:
    """Both models use `class_weight='balanced'`; the paper's reported
    recall of 0.82 on a 16%-positive target is otherwise unreachable, and
    applying it to both keeps the comparison even-handed. Regularisation is
    left at sklearn defaults -- the paper does not state its choice."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, class_weight="balanced",
            random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE),
    }


def _evaluate(estimator, data: PreparedData, cols: list[str]) -> dict:
    """Fit on train, choose the threshold on validation, score test once."""
    model = clone(estimator).fit(data.X_train_t[cols], data.y_train)
    p_val = model.predict_proba(data.X_val_t[cols])[:, 1]
    p_test = model.predict_proba(data.X_test_t[cols])[:, 1]

    thr, val_f1 = best_threshold(data.y_val.to_numpy(), p_val)
    pred_05 = (p_test >= 0.5).astype(int)
    pred_thr = (p_test >= thr).astype(int)

    tuned = point_metrics(data.y_test, pred_thr)
    row = {
        "n_features": len(cols),
        "val_threshold": round(thr, 4),
        "val_f1": round(val_f1, 4),
        "f1_at_0.5": round(f1_score(data.y_test, pred_05, zero_division=0), 4),
        **{k: round(v, 4) for k, v in tuned.items()},
        **{k: round(v, 4) for k, v in ranking_metrics(data.y_test, p_test).items()},
        "f1_ci": bootstrap_metric_ci(data.y_test, pred_thr),
    }
    pr = precision_recall_curve(data.y_test, p_test)
    row["_curve"] = {"proba": p_test, "threshold": thr,
                     "precision": pr[0], "recall": pr[1],
                     "op": point_metrics(data.y_test, pred_thr)}
    return row, model


def run_experiment(verbose: bool = True) -> dict:
    data = prepare_data()
    models = make_models()

    rows, curves, selections, lr_maxout_model, lr_maxout_cols = [], {}, {}, None, None
    for name, est in models.items():
        eliminated, detail = maxout_select(
            data.pool_raw, data.pool_y, est, model_name=name, verbose=verbose)
        selections[name] = detail
        kept = [c for c in data.feature_names if c not in eliminated]

        for variant, cols in [("all_features", data.feature_names),
                              ("maxout", kept)]:
            res, model = _evaluate(est, data, cols)
            curve = res.pop("_curve")
            rows.append({"model": name, "variant": variant, **res})
            if variant == "maxout":
                curves[name] = {
                    "grid": np.linspace(0.02, 0.98, 200),
                    "f1": [f1_score(data.y_test,
                                    (curve["proba"] >= t).astype(int),
                                    zero_division=0)
                           for t in np.linspace(0.02, 0.98, 200)],
                    "thr": curve["threshold"],
                    "precision": curve["precision"], "recall": curve["recall"],
                    "pr_auc": res["pr_auc"],
                    "op_precision": curve["op"]["precision"],
                    "op_recall": curve["op"]["recall"],
                }
            if name == "LogisticRegression" and variant == "maxout":
                lr_maxout_model, lr_maxout_cols = model, cols

    comparison = pd.DataFrame(PAPER_RESULTS + rows)
    comparison.to_csv(TABLE_DIR / "model_comparison.csv", index=False)

    # Paper Section 5.3, on the training rows (Section 3's reading; Section
    # 5.3's "original dataset" wording contradicts it -- noted in the report).
    boot = bootstrap_coefficients(
        make_models()["LogisticRegression"],
        data.X_train_t[lr_maxout_cols], data.y_train, lr_maxout_cols)
    boot.to_csv(TABLE_DIR / "lr_maxout_bootstrap_coefficients.csv", index=False)

    final_coef = (pd.DataFrame({"feature": lr_maxout_cols,
                                "coefficient": lr_maxout_model.coef_[0]})
                  .sort_values("coefficient", key=np.abs, ascending=False)
                  .reset_index(drop=True))
    final_coef.to_csv(TABLE_DIR / "lr_maxout_coefficients.csv", index=False)

    plot_threshold_and_pr(curves, float(data.y_test.mean()))
    plot_bootstrap_coefficients(boot)

    summary = {
        "split_sizes": {
            "train": int(len(data.y_train)), "val": int(len(data.y_val)),
            "test": int(len(data.y_test)),
            "train_pos_rate": round(float(data.y_train.mean()), 4),
            "test_pos_rate": round(float(data.y_test.mean()), 4),
        },
        "maxout_eliminated": {k: v["eliminated"] for k, v in selections.items()},
        "maxout_drop_counts": {k: v["drop_counts"]
                               for k, v in selections.items()},
    }
    (CACHE_DIR / "experiment_results.json").write_text(json.dumps(summary, indent=2))

    return {"comparison": comparison, "bootstrap_coef": boot,
            "final_coef": final_coef, "selections": selections,
            "summary": summary}


if __name__ == "__main__":
    out = run_experiment(verbose=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== SPLIT ===")
    print(pd.DataFrame([out["summary"]["split_sizes"]]).to_string(index=False))
    print("\n=== MODEL COMPARISON ===")
    print(out["comparison"].to_string(index=False))
    print("\n=== max-out eliminated ===")
    for k, v in out["summary"]["maxout_eliminated"].items():
        print(f"  {k}: {v}")
    print("\nFigures: outputs/figures/06_*, 07_*   "
          "Tables: outputs/tables/model_comparison.csv, lr_maxout_*.csv")
