"""
config.py - Central configuration for paths, labels, hyperparameters and seeds.
"""
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_RAW = ROOT / "data" / "raw" / "codemix_sentiment_data.txt"
DATA_PROCESSED = ROOT / "data" / "processed"
CLEAN_CSV = DATA_PROCESSED / "clean.csv"
SPLITS_JSON = DATA_PROCESSED / "splits.json"

# Model / results paths
MODELS_DIR = ROOT / "models"
BEST_MODEL_DIR = MODELS_DIR / "best"
RESULTS_DIR = ROOT / "results"

# Labels
LABELS = ["NEG", "NTL", "POS"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
LABEL_DISPLAY = {"NEG": "Negative", "NTL": "Neutral", "POS": "Positive"}

# Random seeds
SEEDS = [42, 43, 44]
SPLIT_SEED = 42

# Data split
SMALL_DATASET_THRESHOLD = 2000
N_FOLDS = 5
TRAIN_RATIO = 0.80
DEV_RATIO   = 0.10
TEST_RATIO  = 0.10

# Baseline hyperparameters
TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(1, 5),
    min_df=2,
    sublinear_tf=True,
)
SVM_C_GRID = [0.1, 0.5, 1.0, 2.0]

# Transformer hyperparameters
TRANSFORMER_MODELS = [
    "xlm-roberta-base",
    "google/muril-base-cased",
]
MAX_LENGTH = 64
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16 = True

# API
MAX_TEXT_LENGTH = 500
DEFAULT_PORT = 7860
