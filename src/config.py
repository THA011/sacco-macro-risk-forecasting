"""
Central configuration: filesystem paths, SASRA loan-classification thresholds,
sector taxonomy, and the reproducibility seed. Every other module imports from
here rather than hard-coding these values, so the classification logic and
file layout stay in exactly one place.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_EXTERNAL = ROOT_DIR / "data" / "external"
DATA_INTERIM = ROOT_DIR / "data" / "interim"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
REPORTS_FIGURES = ROOT_DIR / "reports" / "figures"
MODELS_DIR = ROOT_DIR / "models_saved"

for _p in (DATA_RAW, DATA_EXTERNAL, DATA_INTERIM, DATA_PROCESSED, REPORTS_FIGURES, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

LEDGER_CSV = DATA_RAW / "simulated_ledger.csv"
MACRO_CSV = DATA_RAW / "simulated_macro.csv"
FEATURES_CSV = DATA_PROCESSED / "model_features.csv"
MODEL_METRICS_JSON = REPORTS_FIGURES / "metrics.json"

# --- Reproducibility ----------------------------------------------------
RANDOM_SEED = 42

# --- SASRA loan classification thresholds (days past due) ---------------
# Bucket boundaries used across SACCOs regulated under SASRA's prudential
# guidelines. Kept as a single ordered mapping so simulate_ledger.py and any
# downstream labeling stay in sync.
SASRA_CLASSIFICATION_THRESHOLDS = [
    ("Normal", 0, 30),
    ("Watch", 31, 90),
    ("Substandard", 91, 180),
    ("Doubtful", 181, 360),
    ("Loss", 361, None),  # None => open-ended upper bound
]

# Ordinal ranking used for "did this account migrate to a worse bucket"
CLASSIFICATION_ORDER = ["Normal", "Watch", "Substandard", "Doubtful", "Loss"]

# At-risk = anything past "Normal". This is the positive class for the
# early-warning classifier.
AT_RISK_CLASSES = ["Watch", "Substandard", "Doubtful", "Loss"]

# --- Sector taxonomy (drives macro-shock exposure weighting) ------------
SECTORS = [
    "transport",       # highest direct fuel-price sensitivity
    "boda_boda",       # highest direct fuel-price sensitivity
    "agriculture",      # fuel for transport/machinery + input costs
    "retail_trade",     # moderate, indirect (logistics costs)
    "salaried_formal",  # lowest direct sensitivity, income more stable
    "other",
]

# Relative sensitivity multiplier applied when simulating how strongly a
# sector's arrears risk responds to a fuel price shock. Illustrative, not a
# calibrated estimate — replace with regression-estimated elasticities once
# real ledger + macro data is available.
SECTOR_FUEL_SENSITIVITY = {
    "transport": 1.00,
    "boda_boda": 0.90,
    "agriculture": 0.65,
    "retail_trade": 0.40,
    "salaried_formal": 0.15,
    "other": 0.30,
}
