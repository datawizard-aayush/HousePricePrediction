from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports" / "figures"
DATASET_PATH = RAW_DATA_DIR / "mumbai_house_prices_cleaned_cr.csv"
MODEL_PATH = ROOT_DIR / "model_pipeline.joblib"
METADATA_PATH = ROOT_DIR / "metadata.json"
DIAGNOSTICS_PATH = ROOT_DIR / "diagnostics.json"

TARGET_COLUMN = "price_in_crores"
FEATURE_COLUMNS = ["region", "bhk", "type", "area", "status", "age"]
NUMERIC_FEATURES = ["bhk", "area"]
CATEGORICAL_FEATURES = ["region", "type", "status", "age"]
