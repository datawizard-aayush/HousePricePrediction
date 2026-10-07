"""Train regression/classification models and generate all dashboard artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.house_price_prediction.classification import (
    save_classification_figures,
    train_classifiers,
)
from src.house_price_prediction.clustering import build_market_segments
from src.house_price_prediction.config import (
    DATASET_PATH,
    DIAGNOSTICS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.house_price_prediction.data_processing import build_metadata, load_clean_data
from src.house_price_prediction.evaluation import train_prediction_interval_models
from src.house_price_prediction.explainability import (
    build_explainability,
    build_group_error_report,
    build_learning_diagnostics,
)
from src.house_price_prediction.modeling import (
    save_model_bundle,
    train_regression,
)

FIGURES_DIR = REPORTS_DIR


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def _save_regression_figures(bundle: dict, results_df: pd.DataFrame, df: pd.DataFrame) -> None:
    diagnostics = bundle["diagnostics"]
    actual = np.asarray(diagnostics["actual_prices"])
    predicted = np.asarray(diagnostics["predicted_prices"])
    residuals = np.asarray(diagnostics["residuals"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(actual, predicted, alpha=0.45, s=12)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0].plot(limits, limits, "r--")
    axes[0].set(title="Actual vs Predicted", xlabel="Actual (Cr)", ylabel="Predicted (Cr)")
    axes[1].scatter(predicted, residuals, alpha=0.45, s=12)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(title="Residuals vs Predicted", xlabel="Predicted (Cr)", ylabel="Residual (Cr)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_diagnostics.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(residuals, bins=45, color="slateblue", alpha=0.8)
    axes[0].set(title="Holdout Residual Distribution", xlabel="Residual (Cr)", ylabel="Frequency")
    axes[1].hist(df[TARGET_COLUMN], bins=45, color="steelblue", alpha=0.8)
    axes[1].set(title="Raw Price Distribution", xlabel="Price (Cr)", ylabel="Listings")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_and_residual_distributions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = results_df.sort_values("CV_MAE_Mean")
    ax.barh(ordered["Model"], ordered["CV_MAE_Mean"], xerr=ordered["CV_MAE_Std"])
    ax.set(title="Regression Model Comparison (5-fold CV)", xlabel="Mean MAE (Cr)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "regression_model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Train the models and write the model bundle, metadata, metrics, and figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    print("Loading and cleaning Mumbai listing data...", flush=True)
    df = load_clean_data(DATASET_PATH)
    print(f"  {len(df):,} rows remain after filters.", flush=True)

    print("Training and comparing regression models...", flush=True)
    bundle, regression_results, X_train, X_test, y_train_log = train_regression(df)
    y_test_cr = df.loc[X_test.index, TARGET_COLUMN]
    y_train_cr = df.loc[X_train.index, TARGET_COLUMN]

    print("Training price-tier classifiers...", flush=True)
    classification = train_classifiers(
        X_train, X_test, y_train_cr, y_test_cr
    )
    save_classification_figures(classification, FIGURES_DIR)

    print("Building region market segments from training data...", flush=True)
    market = build_market_segments(df.loc[X_train.index])
    region_clusters = market["region_clusters"]

    print("Fitting quantile models for prediction intervals...", flush=True)
    interval_models = train_prediction_interval_models(X_train, y_train_log)
    lower, upper = [], []
    for start in range(0, len(X_test), 2000):
        low_pred = np.exp(interval_models["q10"].predict(X_test.iloc[start : start + 2000]))
        high_pred = np.exp(interval_models["q90"].predict(X_test.iloc[start : start + 2000]))
        lower.extend(np.minimum(low_pred, high_pred).tolist())
        upper.extend(np.maximum(low_pred, high_pred).tolist())
    point = np.asarray(bundle["diagnostics"]["predicted_prices"])
    lower = np.minimum(lower, point).tolist()
    upper = np.maximum(upper, point).tolist()

    print("Generating model explainability and bias/variance artifacts...", flush=True)
    explainability = build_explainability(
        bundle["best_pipeline"], X_train, X_test, bundle["test_target_log"]
    )
    learning_diagnostics = build_learning_diagnostics(
        bundle["best_pipeline"], X_train, y_train_log
    )
    group_errors = build_group_error_report(
        X_test,
        bundle["diagnostics"]["actual_prices"],
        bundle["diagnostics"]["predicted_prices"],
        region_clusters,
    )

    bundle.update(classification)
    bundle.update(
        {
            "region_clusters": region_clusters,
            "region_summary": market["region_summary"],
            "cluster_diagnostics": market["cluster_diagnostics"],
            "prediction_interval_models": interval_models,
            "prediction_interval_holdout": {"lower": lower, "upper": upper},
            "explainability": explainability,
            "learning_diagnostics": learning_diagnostics,
            "group_errors": group_errors,
        }
    )
    bundle.pop("classifiers", None)
    bundle.pop("trained_models", None)
    bundle.pop("cluster_model", None)
    bundle.pop("cluster_scaler", None)
    bundle.pop("test_features", None)
    bundle.pop("test_target_log", None)

    model_names = regression_results["Model"].tolist() + list(
        classification["classification_metrics"].keys()
    )
    metadata = build_metadata(df, model_names=model_names, region_clusters=region_clusters)
    with METADATA_PATH.open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, default=_json_default)
    save_model_bundle(bundle, MODEL_PATH)

    _save_regression_figures(bundle, regression_results, df)
    regression_results.to_csv(Path("reports") / "regression_benchmarks.csv", index=False)
    classification_results = pd.DataFrame.from_dict(
        classification["classification_metrics"], orient="index"
    ).drop(columns=["Confusion_Matrix"])
    classification_results.to_csv(Path("reports") / "classification_benchmarks.csv")

    diagnostic_payload = {
        "regression": bundle["diagnostics"],
        "price_distribution": {
            "raw": df[TARGET_COLUMN].astype(float).tolist(),
            "log": np.log(df[TARGET_COLUMN].astype(float)).tolist(),
        },
        "regression_metrics": bundle["metrics"],
        "classification_metrics": classification["classification_metrics"],
        "classification_curves": classification["classification_curves"],
        "classification_holdout": classification["classification_holdout"],
        "price_tier_names": classification["price_tier_names"],
        "cluster_diagnostics": bundle["cluster_diagnostics"],
        "region_summary": bundle["region_summary"],
        "explainability": explainability,
        "learning_diagnostics": learning_diagnostics,
        "group_errors": group_errors,
        "prediction_interval_holdout": bundle["prediction_interval_holdout"],
    }
    with DIAGNOSTICS_PATH.open("w", encoding="utf-8") as output:
        json.dump(diagnostic_payload, output, indent=2, default=_json_default)

    print("\n=== Regression benchmark (Crores) ===")
    print(
        regression_results.to_string(
            index=False,
            formatters={
                "CV_MAE_Mean": lambda value: f"{value:.3f}",
                "CV_MAE_Std": lambda value: f"{value:.3f}",
                "CV_R2_Mean": lambda value: f"{value:.4f}",
                "CV_R2_Std": lambda value: f"{value:.4f}",
                "Test_MAE": lambda value: f"{value:.3f}",
                "Test_RMSE": lambda value: f"{value:.3f}",
                "Test_R2": lambda value: f"{value:.4f}",
            },
        )
    )
    print("\n=== Classification benchmark ===")
    print(classification_results.to_string())
    print(f"\nBest regression model by CV MAE: {bundle['best_model_name']}")
    print(f"Best price-tier classifier: {bundle['best_classifier_name']}")
    print(f"Saved model bundle: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(f"Saved diagnostics: {DIAGNOSTICS_PATH}")
    print(f"Saved figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
