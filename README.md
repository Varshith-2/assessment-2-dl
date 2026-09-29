# Assignment #3 — Improving the Data Science Pipeline

Builds on the Assignment #2 corrected replication of Najafi-Zangeneh et al.,
*"An Improved Machine Learning-Based Employees Attrition Prediction
Framework with Emphasis on Feature Selection"*, Mathematics 2021, 9, 1226
(`mathematics-09-01226-v2.pdf`), on the IBM HR Analytics Employee Attrition
dataset (`data/mainData.csv`).

**Status: Step 1 — baseline established.** The corrected Assignment #2
pipeline (EDA through model evaluation) has been reproduced unchanged in
this project, in a single notebook, as the fixed reference point (E0) for
every later preprocessing/tuning experiment.

## Layout

| Path | Contents |
|---|---|
| `notebooks/00_baseline.ipynb` | **Single notebook, EDA → model evaluation** — the Assignment #3 baseline (E0) |
| `src/` | All pipeline logic (never hand-edited in the notebook) |
| `outputs/figures/`, `outputs/tables/` | Generated artefacts from the baseline run |
| `outputs/cache/` | Cached max-out search + experiment summary (JSON) |
| `data/mainData.csv` | IBM HR Analytics Employee Attrition dataset (1470 × 35) |
| `mathematics-09-01226-v2.pdf` | The paper being replicated and improved on |
| `Assignment-Insturction.md` | The assignment brief |
| `CLAUDE.md` | Guidance for Claude Code working in this repo |

## Source layout (`src/`)

| Path | Contents |
|---|---|
| `config.py` | All constants — seed, split sizes, max-out protocol, paths |
| `data.py` | Load, verify, and the split-first (leakage-safe) splitter |
| `preprocessing.py` | `ColumnTransformer` (standardise + dummy-code), fit on train only |
| `maxout.py` | The paper's 1-max-out feature selection, leakage-safe, cached |
| `evaluation.py` | Metrics, validation-tuned threshold, bootstrap CIs + coefficients |
| `experiment.py` | Orchestration: fair 2×2 comparison + Section 5.3 bootstrap analysis |
| `eda.py` | EDA analyses (return tidy DataFrames, write CSVs) |
| `plotting.py` | Shared theme + palette + every figure builder |
| `build_notebook.py` | Regenerates `notebooks/00_baseline.ipynb` from `src/` |

## Reproduce the baseline

```sh
python3 -m venv .venv
.venv/bin/pip install pandas numpy scikit-learn matplotlib seaborn scipy \
    statsmodels jupyterlab nbformat nbconvert ipykernel markdown

cd src
../.venv/bin/python eda.py               # 9 tables, figures 1–5
../.venv/bin/python experiment.py        # comparison table, figures 6–7 (~11 min: max-out search)
../.venv/bin/python build_notebook.py    # (re)generate notebooks/00_baseline.ipynb
cd ../notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 00_baseline.ipynb
```

The max-out search caches to `outputs/cache/maxout_*.json`, keyed by a
signature of the model configuration and protocol. Delete those files to
force a re-run; a changed model config invalidates them automatically.

## Baseline (E0) — Assignment #2's corrected replication

Test-set results at a validation-tuned decision threshold, `class_weight="balanced"`:

| Model | Variant | Acc | Prec | Rec | F1 | ROC-AUC | PR-AUC | n_features |
|---|---|---|---|---|---|---|---|---|
| LogisticRegression | all_features | 0.824 | 0.457 | 0.444 | 0.451 | 0.835 | 0.603 | 52 |
| LogisticRegression | maxout | 0.833 | 0.485 | 0.444 | 0.464 | 0.833 | 0.604 | 44 |
| RandomForest | all_features | 0.810 | 0.429 | 0.500 | 0.462 | 0.805 | 0.489 | 52 |
| RandomForest | maxout | 0.851 | 0.552 | 0.444 | 0.492 | 0.828 | 0.513 | 50 |

F1 is the primary metric (16.1% positive class). The bootstrap CI on test F1
spans ≈ 0.30 on the 221-row test set — single-split test deltas smaller than
that are not resolvable; later experiments compare under shared stratified
k-fold CV instead.

**Two methodological corrections kept from Assignment #2:**

1. **Split first.** All scaling, encoding, and feature selection are fit on
   training rows only; the test set is scored once.
2. **Fair model comparison.** LR and RF run on one identical split, seed,
   and preprocessor, each with its own max-out search, each scored at a
   validation-tuned decision threshold, with bootstrap CIs on test F1 and
   threshold-free ROC-AUC / PR-AUC alongside.

## Next steps (Assignment #3, beyond the baseline)

1. Formulate hypotheses from the EDA findings (see `notebooks/00_baseline.ipynb` Part 1).
2. Evaluate preprocessing improvements one at a time under shared stratified k-fold CV.
3. Tune the main model's hyperparameters under the same CV protocol, without touching the test set.
4. Score the final pipeline against this baseline on the held-out test set.
