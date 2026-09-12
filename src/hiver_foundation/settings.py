"""Project-wide settings. Change values here; keep RANDOM_SEED fixed for reproducibility."""

from pathlib import Path

# Repo root: src/hiver_foundation/settings.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]

RANDOM_SEED = 42

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
DATA_GOLDEN_DIR = ROOT / "data" / "golden"
REPORTS_DIR = ROOT / "reports"
TAXONOMY_DIR = ROOT / "configs"

# Primary dataset from the assignment:
# Kaggle: thoughtvector/customer-support-on-twitter  (file: twcs.csv)
DEFAULT_RAW_CSV = DATA_RAW_DIR / "twcs.csv"
SAMPLE_RAW_CSV = DATA_RAW_DIR / "sample.csv"

# Conversation-level split. These three must sum to 1.0.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Prefer a configured brand if it exists in the loaded file.
# Leave as None to auto-select with a deterministic rule (see brands.py).
PREFERRED_BRAND = None

# Ignore tiny brands so auto-select stays usable on a subsample.
MIN_CONVERSATIONS_FOR_BRAND = 30

# Golden-set sample sizes (candidates for HAND labeling). Drawn after the split
# so train candidates never come from test conversations.
GOLDEN_TRAIN_N = 150
GOLDEN_VAL_N = 50
GOLDEN_TEST_N = 50

# Sklearn TF-IDF / logistic regression
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 200
