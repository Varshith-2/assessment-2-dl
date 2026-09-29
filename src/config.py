"""Shared configuration for the Assignment #2 EDA + replication pipeline."""
from pathlib import Path

# One seed threaded through every split, model, and bootstrap so the whole
# study is reproducible end to end.
RANDOM_STATE = 42

# Split sizes. The paper describes a train / validation / test protocol but
# never states the ratios; 70 / 15 / 15 (stratified) keeps an explicit
# validation set as the paper's 1-max-out procedure requires.
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the FULL dataset; converted to a fraction of
                 # the train+val remainder inside the splitter.

# 1-max-out protocol, taken from the paper (Section 5.1):
#   - repeat the greedy backward elimination over N_MAXOUT_RUNS random
#     train / validation splits
#   - a feature is eliminated if it is dropped in more than
#     MAXOUT_ELIM_THRESHOLD of those runs.
N_MAXOUT_RUNS = 7
MAXOUT_ELIM_THRESHOLD = 4

# Post-processing coefficient-confidence analysis (paper Section 5.3).
N_BOOTSTRAP = 300

# Bootstrap CIs on the held-out test metrics (our addition, so model
# comparison is a claim with error bars rather than a single number).
N_TEST_BOOTSTRAP = 2000

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "mainData.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"

for _d in (FIGURE_DIR, TABLE_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

TARGET = "Attrition"

# Columns that are constant across all rows. The paper names StandardHours
# and EmployeeCount and omits them BEFORE feature selection, so we do the
# same. Over18 is ALSO constant, but the paper leaves it in until selection
# removes it (its "Over18_yes" dummy appears in the paper's omitted list) --
# we mirror that so the replication stays faithful; the EDA flags it.
ALL_CONSTANT_COLS = ["StandardHours", "EmployeeCount", "Over18"]
PAPER_DROPPED_CONSTANTS = ["StandardHours", "EmployeeCount"]

# Pure row identifier, no predictive meaning. The paper never flags it.
ID_COL = "EmployeeNumber"

# 1-4 / 1-5 ordinal survey scales the paper treats as numeric (inherited
# assumption, flagged in the EDA).
ORDINAL_NUMERIC_COLS = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel",
    "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "WorkLifeBalance",
]
