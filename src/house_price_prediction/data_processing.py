from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def normalize_price_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Convert supported source price formats into Crores."""
    processed = df.copy()
    if "price_cr" in processed.columns:
        processed[TARGET_COLUMN] = pd.to_numeric(processed["price_cr"], errors="coerce")
    elif {"price", "price_unit"}.issubset(processed.columns):
        processed["price_unit"] = processed["price_unit"].astype(str).str.strip().str.upper()
        processed["price"] = pd.to_numeric(processed["price"], errors="coerce")
        processed[TARGET_COLUMN] = np.where(
            processed["price_unit"] == "CR",
            processed["price"],
            np.where(processed["price_unit"] == "L", processed["price"] / 100.0, np.nan),
        )
    else:
        raise ValueError("Dataset must contain 'price_cr' or both 'price' and 'price_unit'.")

    return processed


def clean_region_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize region strings and group rare micro-markets into 'Other'."""
    processed = df.copy()
    processed["region"] = processed["region"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

    region_counts = processed["region"].value_counts()
    rare_regions = region_counts[region_counts < 5].index
    processed["region"] = processed["region"].where(~processed["region"].isin(rare_regions), "Other")
    return processed


def clean_and_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project-specific data cleaning rules for Mumbai property listings."""
    processed = normalize_price_unit(df)
    processed = clean_region_values(processed)

    processed["bhk"] = pd.to_numeric(processed["bhk"], errors="coerce")
    processed["area"] = pd.to_numeric(processed["area"], errors="coerce")
    processed["type"] = processed["type"].astype(str).str.strip()
    processed["status"] = processed["status"].astype(str).str.strip()
    processed["age"] = processed["age"].astype(str).str.strip()
    processed["age"] = processed["age"].replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})

    processed = processed.dropna(subset=[TARGET_COLUMN]).copy()
    processed = processed[(processed["area"] >= 150) & (processed["area"] <= 8000)]
    processed = processed[(processed["bhk"] >= 1) & (processed["bhk"] <= 6)]

    processed["price_per_sqft"] = (processed[TARGET_COLUMN] * 10000000.0) / processed["area"]
    processed = processed[(processed["price_per_sqft"] >= 2500) & (processed["price_per_sqft"] <= 120000)]

    processed = processed[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    return processed.reset_index(drop=True)


def load_clean_data(path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the real dataset and return the filtered, cleaned modelling frame."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")

    raw_df = pd.read_csv(file_path)
    required_columns = {"bhk", "type", "locality", "area", "region", "status", "age"}
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if "price_cr" not in raw_df.columns and not {"price", "price_unit"}.issubset(raw_df.columns):
        raise ValueError("Dataset must contain 'price_cr' or both 'price' and 'price_unit'.")

    cleaned = clean_and_prepare_dataframe(raw_df)
    return cleaned


def build_metadata(df: pd.DataFrame) -> dict:
    metadata = {
        "regions": sorted(df["region"].dropna().unique().tolist()),
        "property_types": sorted(df["type"].dropna().unique().tolist()),
        "statuses": sorted(df["status"].dropna().unique().tolist()),
        "age_categories": sorted(df["age"].dropna().unique().tolist()),
    }
    return metadata
