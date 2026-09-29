"""The paper's 'max-out' feature-selection method (m = 1: greedy backward
elimination), reimplemented leakage-safe.

Paper, Section 5.1: the test set is held out first. Then, over 7 iterations,
the train/validation boundary is re-randomised and a full greedy 1-max-out
run is performed. A feature is eliminated if it is dropped in more than 4 of
the 7 runs. Each run scores candidate feature sets by validation F1.

Here every run re-fits the preprocessor on its own inner-training rows, so
scaling and encoding never see the inner-validation rows.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import CACHE_DIR, MAXOUT_ELIM_THRESHOLD, N_MAXOUT_RUNS, RANDOM_STATE
from preprocessing import build_preprocessor, fit_transform_frame, transform_frame


def _greedy_one_maxout(estimator, Xtr, ytr, Xva, yva, verbose=False):
    """One greedy 1-max-out pass. Returns the ordered list of removed
    feature names. Stops when no single removal improves validation F1."""
    active = list(Xtr.columns)

    def score(cols):
        est = clone(estimator)
        est.fit(Xtr[cols], ytr)
        return f1_score(yva, est.predict(Xva[cols]))

    best = score(active)
    removed = []
    while len(active) > 1:
        candidate_scores = []
        for f in active:
            trial = [c for c in active if c != f]
            candidate_scores.append((score(trial), f))
        cand_best, cand_feat = max(candidate_scores, key=lambda t: t[0])
        # Paper: stop only once the metric drops below the previous step.
        if cand_best >= best:
            best = cand_best
            active.remove(cand_feat)
            removed.append(cand_feat)
            if verbose:
                print(f"  drop {cand_feat!r:40s} val F1 -> {best:.4f} "
                      f"({len(active)} left)")
        else:
            break
    return removed


def maxout_select(
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    estimator,
    *,
    model_name: str,
    n_runs: int = N_MAXOUT_RUNS,
    threshold: int = MAXOUT_ELIM_THRESHOLD,
    inner_val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    cache: bool = True,
    verbose: bool = True,
):
    """Run the full 7-iteration max-out procedure on the train+validation
    pool (raw features). Returns (eliminated_features, detail_dict).

    `estimator` must be an unfitted sklearn classifier; it is cloned for
    every fit. `model_name` keys the on-disk cache so the expensive search
    is not repeated between notebook runs.
    """
    # Signature of everything the search result depends on. Stored in the
    # cache so a changed model config / protocol is not silently reused.
    signature = "|".join([
        repr(estimator), str(n_runs), str(threshold), str(inner_val_size),
        str(random_state), str(list(X_pool.columns)), str(len(X_pool)),
    ])
    cache_path = Path(CACHE_DIR) / f"maxout_{model_name}.json"
    if cache and cache_path.exists():
        detail = json.loads(cache_path.read_text())
        if detail.get("signature") == signature:
            if verbose:
                print(f"[maxout:{model_name}] loaded cache {cache_path.name}")
            return detail["eliminated"], detail
        if verbose:
            print(f"[maxout:{model_name}] cache signature stale -- re-running")

    rng = np.random.RandomState(random_state)
    per_run_removed = []
    for run in range(n_runs):
        rs = int(rng.randint(0, 2**31 - 1))
        Xtr_raw, Xva_raw, ytr, yva = train_test_split(
            X_pool, y_pool, test_size=inner_val_size, stratify=y_pool,
            random_state=rs,
        )
        pre = build_preprocessor(Xtr_raw)
        Xtr = fit_transform_frame(pre, Xtr_raw)
        Xva = transform_frame(pre, Xva_raw)
        if verbose:
            print(f"[maxout:{model_name}] run {run + 1}/{n_runs} "
                  f"(seed {rs})")
        removed = _greedy_one_maxout(estimator, Xtr, ytr, Xva, yva,
                                     verbose=verbose)
        per_run_removed.append(removed)

    counts = Counter()
    for r in per_run_removed:
        counts.update(set(r))
    eliminated = sorted(f for f, c in counts.items() if c > threshold)

    detail = {
        "model_name": model_name,
        "signature": signature,
        "n_runs": n_runs,
        "threshold": threshold,
        "inner_val_size": inner_val_size,
        "random_state": random_state,
        "per_run_removed": per_run_removed,
        "drop_counts": dict(counts.most_common()),
        "eliminated": eliminated,
    }
    if cache:
        cache_path.write_text(json.dumps(detail, indent=2))
        if verbose:
            print(f"[maxout:{model_name}] wrote cache {cache_path.name}")
    return eliminated, detail
