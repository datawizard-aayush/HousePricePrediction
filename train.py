from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.house_price_prediction.config import DATASET_PATH, METADATA_PATH, MODEL_PATH
from src.house_price_prediction.data_processing import build_metadata, load_clean_data
from src.house_price_prediction.modeling import save_model_bundle, train_and_evaluate


def main() -> None:
    """Training entry point for the Mumbai house price prediction project."""
    dataset_path = Path(DATASET_PATH)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = load_clean_data(dataset_path)
    bundle, metrics_df = train_and_evaluate(df)

    save_model_bundle(bundle)

    metadata = build_metadata(df)
    with METADATA_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print("\n=== Mumbai House Price Prediction Benchmark ===")
    print(metrics_df.to_string(index=False, formatters={
        "MAE": lambda value: f"₹{value:,.2f} Cr",
        "RMSE": lambda value: f"₹{value:,.2f} Cr",
        "R2": lambda value: f"{value:.4f}",
    }))
    print(f"\nBest model selected: {bundle['best_model_name']}")
    print(f"Saved model pipeline: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()
