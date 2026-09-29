"""Generate the single baseline notebook from the src/ modules.

The notebook stays thin -- import, call, display, narrate -- while every
computation lives in a version-controlled .py file. Re-run this script
whenever the analysis code changes, then execute the notebook with
`jupyter nbconvert --execute`.

This assembles ONE notebook (EDA through model evaluation) rather than the
two separate notebooks Assignment #2 used, per the Assignment #3 baseline
step: reproduce the corrected A2 pipeline end to end as a single artefact.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

BOOT = (
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path.cwd().parent / 'src'))\n"
    "import pandas as pd\n"
    "pd.set_option('display.max_columns', 60)\n"
    "pd.set_option('display.width', 170)\n"
)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


cells = [
    md("# Assignment #3 -- Baseline: EDA and Corrected Replication\n\n"
       "**Paper.** Najafi-Zangeneh, Shams-Gharneh, Arjomandi-Nezhad & "
       "Hashemkhani Zolfani, *An Improved Machine Learning-Based Employees "
       "Attrition Prediction Framework with Emphasis on Feature Selection*, "
       "Mathematics 2021, 9, 1226.\n\n"
       "**Dataset.** IBM HR Analytics Employee Attrition "
       "(`data/mainData.csv`), 1470 rows x 35 columns.\n\n"
       "**Purpose.** Step 1 of Assignment #3 -- establish the baseline. "
       "This notebook reproduces the Assignment #2 *corrected* replication "
       "(EDA through model evaluation) unchanged: same preprocessing, "
       "features, model configuration, evaluation metrics, split, and "
       "procedure. Every subsequent Assignment #3 experiment is compared "
       "against the numbers produced here. All computation lives in "
       "`src/`; this notebook only imports, calls, displays, and narrates."),
]

# ====================================================================== EDA
cells += [
    md("## Part 1 -- Exploratory Data Analysis\n\n"
       "Each section answers one question that feeds a decision in Part 2. "
       "All computation is in `src/eda.py`; figures are built by "
       "`src/plotting.py`."),
    code(BOOT + "import eda\nfrom data import load_raw\n"
                "df = load_raw()\ntables = eda.run_all(df)\ndf.head()"),

    md("### 1.1 Dataset verification against the paper\n\n"
       "Observation count, feature count and types, target, missing values, "
       "duplicates, class distribution - and every difference from the "
       "paper's description."),
    code("tables['verification']"),
    md("**Findings.**\n\n"
       "- Row and column counts match the paper exactly (1470 x 35).\n"
       "- **Three** columns are constant, not two: the paper names "
       "`StandardHours` and `EmployeeCount` but never mentions `Over18` "
       "(constant `'Y'`). We drop the two the paper drops and let feature "
       "selection remove the `Over18` dummy, as the paper implicitly does.\n"
       "- The paper lists `EmployeeNumber` among the predictive features; it "
       "is a pure row identifier, so we drop it (31 modelling features vs "
       "the paper's 32).\n"
       "- **No missing cells and no duplicate rows**, even after ignoring "
       "`EmployeeNumber` - so no imputation or de-duplication step is "
       "needed.\n"
       "- `Attrition` is binary and imbalanced."),

    md("### 1.2 Class imbalance\n\n"
       "The single most consequential property of the dataset for both "
       "modelling and evaluation."),
    code("tables['class_counts']"),
    md("![class balance](../outputs/figures/01_class_balance.png)\n\n"
       "**Finding.** Only **16.1%** of employees left (237 of 1470). A "
       "model that always predicts \"stays\" scores 83.9% accuracy while "
       "being useless, so accuracy cannot be the headline metric - the "
       "paper argues the same and reports precision / recall / F1. This "
       "also motivates `class_weight='balanced'` in Part 2: without it the "
       "minority-class recall collapses and the paper's reported recall of "
       "0.82 is unreachable."),

    md("### 1.3 Relationship between features and the target"),
    code("tables['numeric_association']"),
    md("![numeric association](../outputs/figures/"
       "02_numeric_target_association.png)"),
    code("tables['categorical_association']"),
    md("![categorical association](../outputs/figures/"
       "03_categorical_target_association.png)\n\n"
       "**Findings.** Leavers are younger, have fewer total working years, "
       "shorter tenure, lower job level and income, and live farther from "
       "the office (all |Cohen's d| ~ 0.2-0.5). Among categoricals the "
       "strongest signal is **`OverTime`** (31% vs 10%), then `JobRole` "
       "(Sales Representative ~40%), frequent business travel (25%), and "
       "`MaritalStatus` Single (26%). These directions match domain "
       "expectation and give a sanity check for Part 2: a feature-selection "
       "result that discards `OverTime` or `TotalWorkingYears` would be "
       "suspect."),

    md("### 1.4 Correlation and redundancy among features\n\n"
       "The paper reads logistic-regression coefficients directly as "
       "\"feature importance\". That is only safe if the features are not "
       "badly collinear."),
    code("tables['high_correlations']"),
    code("tables['vif'].head(12)"),
    md("![correlation heatmap](../outputs/figures/"
       "04_correlation_heatmap.png)\n\n"
       "**Findings.** `JobLevel` and `MonthlyIncome` are near-duplicates "
       "(r = 0.95; VIF ~ 11 each). `TotalWorkingYears` correlates with "
       "both, and the four `Years*` tenure columns form a second correlated "
       "block (r = 0.6-0.8). **Implication for Part 2:** individual "
       "logistic-regression coefficients in this space are unstable - a "
       "small change in the training sample can shift weight between "
       "`JobLevel` and `MonthlyIncome` - so the paper's "
       "coefficient-as-importance narrative needs caution, and the "
       "bootstrap coefficient-stability analysis (paper Section 5.3) is "
       "exactly the right diagnostic to keep. This is also the strongest "
       "candidate for an Assignment #3 feature-redundancy experiment."),

    md("### 1.5 Are the ordinal survey scales really numeric?\n\n"
       "The paper feeds 1-4 / 1-5 satisfaction and involvement ratings to "
       "the model as continuous numbers. We inherit that; here is what is "
       "being assumed."),
    code("tables['ordinal_scales']"),
    md("**Finding.** Nine features are discrete ratings with 2-5 levels. "
       "Treating them as numeric assumes equal spacing between levels - a "
       "documented assumption carried from the paper, not an independent "
       "choice. `PerformanceRating` has only two observed values (3, 4) and "
       "is effectively binary. Another candidate Assignment #3 experiment: "
       "encode these as categorical instead of numeric."),

    md("### 1.6 Outliers and anomalous observations"),
    code("tables['outliers']"),
    md("![outliers](../outputs/figures/05_outliers_strip.png)\n\n"
       "**Findings.** The monetary and tenure columns are right-skewed with "
       "long upper tails: ~7-8% of `MonthlyIncome`, `YearsAtCompany` and "
       "`YearsSinceLastPromotion` sit beyond the 1.5x IQR fence. Split by "
       "attrition, the tails are not concentrated among leavers - leavers "
       "are simply shifted toward lower values overall. These are plausible "
       "values (long-serving senior staff), not data errors, so we keep "
       "them and rely on standardisation; the paper applies no outlier "
       "treatment and we match that."),

    md("### 1.7 Data-leakage audit"),
    code("tables['leakage']"),
    md("**Findings.** `EmployeeNumber` is an identifier and is dropped; the "
       "constant columns carry no information. No feature is measured "
       "*after* the attrition outcome, so there is no target leakage inside "
       "the feature set. The dataset is a single cross-sectional snapshot "
       "with no timestamp, so the brief's time-series checks (trend, "
       "seasonality, missing timestamps, autocorrelation) **do not apply**. "
       "The real leakage risk here is **procedural**: fitting scaling or "
       "feature selection before the train/test split - which is exactly "
       "what Part 2 corrects, and which every Assignment #3 experiment must "
       "also preserve (fit inside the fold, never before it)."),

    md("### EDA summary and implications for the baseline\n\n"
       "| # | Finding | Implication |\n"
       "|---|---|---|\n"
       "| 1 | 1470 x 35, no missing values, no duplicates | Clean start; no "
       "imputation / de-duplication |\n"
       "| 2 | `Over18` is a third constant; paper counts `EmployeeNumber` "
       "as a feature | Drop `EmployeeNumber`; let selection remove `Over18` "
       "(documented deviations) |\n"
       "| 3 | Target only 16.1% positive | Report P/R/F1 not accuracy; "
       "`class_weight='balanced'` for every model; tune the decision "
       "threshold |\n"
       "| 4 | `OverTime`, tenure, `TotalWorkingYears`, income are the "
       "strongest signals | Sanity check for feature selection |\n"
       "| 5 | `JobLevel` ~ `MonthlyIncome` (r 0.95, VIF ~ 11); `Years*` "
       "block correlated | Coefficient-as-importance is fragile; keep the "
       "bootstrap stability analysis |\n"
       "| 6 | 9 ordinal scales treated as continuous | Inherited "
       "equal-spacing assumption; `PerformanceRating` near-binary |\n"
       "| 7 | ~7-8% right-tail outliers, all plausible, not tied to "
       "attrition | Keep rows, standardise, no special treatment |\n"
       "| 8 | Cross-sectional, no time dimension | Time-series checks N/A; "
       "leakage fix is procedural - split before preprocessing |\n"),
]

# ============================================================ replication
cells += [
    md("## Part 2 -- Baseline Pipeline: Replicate and Correct the Paper's "
       "Methodology\n\n"
       "**Paper's pipeline.** data -> dummy-code categoricals -> "
       "\"max-out\" (1-max-out = greedy backward) feature selection -> "
       "logistic regression -> accuracy / precision / recall / F1 -> "
       "300-bootstrap coefficient-confidence analysis.\n\n"
       "**Paper's reported test results (Section 5.2).** accuracy 0.81, "
       "precision 0.43, recall 0.82, F1 0.56 with max-out; "
       "0.78 / 0.39 / 0.82 / 0.53 without it.\n\n"
       "Logic lives in `src/`: `data.py` (split-first), `preprocessing.py` "
       "(fit on train only), `maxout.py` (leakage-safe feature search), "
       "`evaluation.py` (metrics, threshold, bootstrap), `experiment.py` "
       "(orchestration), `plotting.py` (figures)."),

    md("### 2.1 The paper's pipeline, and where it is unclear or wrong\n\n"
       "| Stage | What the paper does | Problem | Our correction |\n"
       "|---|---|---|---|\n"
       "| Preprocessing scope | Section 5.1 splits train+val / test inside "
       "the feature search, but the ordinal scaling and the Section 5.3 "
       "bootstrap are described on the whole dataset | Section 5.3 says the "
       "bootstrap uses \"the original dataset\"; Section 3 says the "
       "training set - a contradiction. The scaling choice is never "
       "stated. | **Split first** (stratified 70/15/15). Fit scaler + "
       "encoder on train only. Bootstrap the **training** set. |\n"
       "| Feature selection | 1-max-out over 7 random train/val splits; "
       "drop features removed > 4 times | Sound, but the preprocessor is "
       "fit once on all of train+val before the 7 splits, leaking "
       "validation statistics into each run | Re-fit the preprocessor on "
       "each run's inner-training rows |\n"
       "| Model comparison | LR + max-out compared against RF / KNN / SVM "
       "**taken from other papers** with different splits, preprocessing "
       "and metrics (Figure 6) | Not a controlled experiment; it cannot "
       "show that LR, or max-out, is the better choice | Run **LR and RF "
       "on one identical split / seed / preprocessor**, each with its own "
       "max-out search (RF selects with RF as the wrapper) |\n"
       "| Operating point | recall 0.82 at 16% prevalence reported with no "
       "mention of threshold or weighting | Unreproducible; and "
       "`class_weight='balanced'` shifts LR's boundary but not RF's "
       "majority vote, so a fixed 0.5 cut compares them unfairly | Apply "
       "`class_weight='balanced'` to **both**; choose each model's "
       "threshold by maximising **validation** F1; also report "
       "threshold-free ROC-AUC / PR-AUC |\n"
       "| Split ratios / seed / regularisation | Not stated | Not "
       "reproducible | Assume 70/15/15, `random_state=42`, sklearn default "
       "L2; all in `src/config.py` |\n"),

    code(BOOT + "import experiment\n"
                "out = experiment.run_experiment(verbose=False)\n"
                "pd.DataFrame([out['summary']['split_sizes']])"),
    md("The 1-max-out search is cached under `outputs/cache/maxout_*.json` "
       "(keyed by a signature of the model config and protocol) so it is "
       "not repeated on every run."),

    md("### 2.2 Correction A - split first, then everything else\n\n"
       "`data.three_way_split` does a stratified 70/15/15 split on the raw "
       "frame. `preprocessing.build_preprocessor` builds a "
       "`ColumnTransformer` (StandardScaler + OneHotEncoder keeping all "
       "levels, matching the paper's dummy coding) fit on training rows "
       "only. Validation and test are transformed with those frozen "
       "statistics; the test set is scored exactly once."),

    md("### 2.3 Feature selection - max-out, leakage-safe\n\n"
       "`maxout.maxout_select` reimplements 1-max-out: greedily drop the "
       "single feature whose removal most improves validation F1, stop when "
       "no removal helps. Repeated over 7 random inner train/val splits; a "
       "feature is eliminated if dropped in more than 4 runs. Each run "
       "re-fits the scaler/encoder on its own inner-training rows."),
    code("for model, elim in out['summary']['maxout_eliminated'].items():\n"
         "    print(f'{model}: {len(elim)} eliminated')\n"
         "    for f in elim:\n"
         "        print('   -', f)"),
    code("pd.Series(out['summary']['maxout_drop_counts']"
         "['LogisticRegression'], name='times_dropped_over_7_runs').head(15)"),
    md("**Comparison with the paper.** The paper eliminated 21 dummy "
       "features; our leakage-safe LR run is more conservative (8). "
       "Reassuringly, **every** feature our LR run eliminates - `JobLevel`, "
       "`MonthlyRate`, `PerformanceRating`, `Gender_Female`, `Education`, "
       "`Department_Research & Development`, `EducationField_Human "
       "Resources`, `EducationField_Marketing` - also appears in the "
       "paper's omitted list, and they are precisely the collinear / "
       "low-signal dummies the EDA flagged. Random Forest eliminates a "
       "different, smaller set (`DailyRate`, `YearsAtCompany`) because it "
       "uses a different wrapper metric - which is the point of running the "
       "search per model."),

    md("### 2.4 Correction B - a fair model comparison\n\n"
       "One split, one seed, one preprocessor. Both models get "
       "`class_weight='balanced'`. Each is evaluated with all features and "
       "with its own max-out selection, at a decision threshold chosen on "
       "the validation set. 95% CIs on test F1 are 2000-resample "
       "percentile bootstraps of the test set."),
    code("cmp = out['comparison']\n"
         "cmp[['model','variant','n_features','val_threshold','f1_at_0.5',"
         "'accuracy','precision','recall','f1','f1_ci','roc_auc','pr_auc']]"),
    md("![fair comparison](../outputs/figures/"
       "06_fair_model_comparison.png)"),

    md("### 2.5 Results vs the paper -- this is the Assignment #3 baseline "
       "(E0)\n\n"
       "| Model | Paper | Baseline (threshold tuned on validation) |\n"
       "|---|---|---|\n"
       "| LR + max-out | acc 0.81, P 0.43, R 0.82, **F1 0.56** | "
       "acc 0.83, P 0.48, R 0.44, **F1 0.46**  (95% CI 0.31-0.60) |\n"
       "| LR, all features | acc 0.78, P 0.39, R 0.82, **F1 0.53** | "
       "acc 0.82, P 0.46, R 0.44, **F1 0.45**  (95% CI 0.30-0.58) |\n"
       "| RF + max-out | not run on a comparable basis | "
       "acc 0.85, P 0.55, R 0.44, **F1 0.49**  (95% CI 0.33-0.63) |\n"
       "| RF, all features | not run on a comparable basis | "
       "acc 0.81, P 0.43, R 0.50, **F1 0.46**  (95% CI 0.31-0.59) |\n\n"
       "**This LR + max-out row (acc 0.83 / P 0.48 / R 0.44 / F1 0.46) is "
       "the Assignment #3 baseline (E0)** that every later experiment (E1, "
       "E2, ... ) is compared against, using the same preprocessing, "
       "features, model configuration, split, and evaluation procedure.\n\n"
       "### Discussion\n\n"
       "**Could we reproduce the reported performance?** Not the headline "
       "numbers. Our LR + max-out reaches F1 ~ 0.46 against the paper's "
       "0.56, and recall 0.44 against 0.82. The paper's combination of "
       "recall 0.82 with precision 0.43 is the profile of a model run at a "
       "**very low decision threshold** (recall bought at the cost of "
       "precision). At a fixed 0.5 cut our LR actually scores higher "
       "(`f1_at_0.5` ~ 0.51); maximising *validation* F1 moves the "
       "threshold the other way, trading recall for precision. Forcing "
       "recall ~ 0.82 on our split drives precision below 0.30. The most "
       "likely explanations for the gap: the paper never fixes a "
       "threshold, regularisation strength, or split seed, and its "
       "Section 5.3 wording suggests some steps touched the whole dataset.\n\n"
       "**Did max-out actually help?** Marginally, and not "
       "distinguishably. LR: F1 0.45 -> 0.46; RF: 0.46 -> 0.49. The "
       "bootstrap CIs overlap almost completely in both cases, so the "
       "improvement the paper attributes to max-out (F1 0.53 -> 0.56) "
       "**cannot be called statistically significant** in a leakage-safe "
       "setting - consistent with how small the paper's own delta is. What "
       "max-out does deliver is a smaller model (44 vs 52 features for LR) "
       "at no measurable cost.\n\n"
       "**LR + max-out vs RF + max-out** - the comparison the paper "
       "should have made. At the tuned operating point RF edges LR on F1 "
       "(0.49 vs 0.46), but the CIs (0.33-0.63 vs 0.31-0.60) overlap "
       "heavily. The threshold-free view is clearer and favours **LR**: "
       "PR-AUC 0.60 vs 0.51, ROC-AUC 0.83 vs 0.83. LR ranks positives "
       "better across all thresholds; RF only pulls ahead at one tuned "
       "point. Net: no strong evidence either model dominates, with a mild "
       "edge to LR as the more robust ranker - consistent with keeping LR "
       "as the main model for Assignment #3 (per the brief, model capacity "
       "is not what is being improved).\n\n"
       "**Did the methodological corrections change the outcome?** Yes. "
       "(1) The leakage-safe split plus a single honest test evaluation "
       "drops F1 from the paper's 0.56 to ~0.46. (2) The fixed-0.5-cut "
       "comparison is actively misleading here: at 0.5, RF + max-out scores "
       "F1 0.19 (`f1_at_0.5`) purely because `class_weight` never moved its "
       "vote threshold - tuning it on validation lifts the same model to "
       "0.49. Any imbalanced-data comparison at a fixed 0.5 is unreliable.\n\n"
       "**Limitations (carried into Assignment #3).** A single 70/15/15 "
       "split for this baseline table (no repeated CV for the headline "
       "number here -- Assignment #3 introduces stratified k-fold CV for "
       "all later experiments); the 221-row test set makes every CI wide "
       "(~0.30 F1 span); the paper's unspecified hyperparameters force "
       "assumptions; `class_weight='balanced'` and the F1-optimal "
       "threshold are our choices, applied equally to both models; the "
       "ordinal-as-numeric encoding is inherited from the paper and is a "
       "candidate for improvement."),

    md("### 2.6 Post-processing - coefficient-confidence analysis (Section "
       "5.3)\n\n"
       "300 bootstrap resamples of the **training** set, LR + max-out refit "
       "on each, reporting the mean, SD and coefficient of variation "
       "(SD / |mean|) of every coefficient. Low CV = stable estimate. The "
       "estimator bootstrapped here is the same object used for the "
       "headline result."),
    code("out['bootstrap_coef']"),
    code("out['final_coef'].head(12)   # single-fit coefficients, "
         "largest magnitude first"),
    md("![bootstrap coefficients](../outputs/figures/"
       "07_bootstrap_coefficients.png)\n\n"
       "**Reading.** The largest, most stable coefficients (CV < 0.5) are "
       "`OverTime`, `BusinessTravel`, `JobSatisfaction`, "
       "`EnvironmentSatisfaction`, `JobRole_Laboratory Technician`, "
       "`MaritalStatus_Single` - and they line up with the EDA's "
       "feature-target section, so these are the signals to trust. The "
       "unstable ones (CV >= 0.5) - `TotalWorkingYears`, `Department_Sales`, "
       "`JobRole_Sales Representative`, `Age` - sit inside the high-VIF / "
       "collinear blocks the EDA identified in section 1.4. This is direct "
       "evidence for the caution raised there: the paper's practice of "
       "reading individual coefficients as feature importance is fragile "
       "for exactly those features, and the bootstrap makes the fragility "
       "visible."),

    md("## Next steps (outside this notebook)\n\n"
       "With E0 established above, later Assignment #3 notebooks/scripts "
       "will: (1) formulate hypotheses from the EDA findings in Part 1, "
       "(2) evaluate preprocessing improvements one at a time under shared "
       "stratified k-fold CV, (3) tune the main model's hyperparameters "
       "under the same CV protocol without touching the test set, and "
       "(4) score the final pipeline against this baseline on the held-out "
       "test set."),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (.venv)",
                   "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

path = NB_DIR / "00_baseline.ipynb"
nbf.write(nb, path)
print("wrote", path.relative_to(ROOT))
