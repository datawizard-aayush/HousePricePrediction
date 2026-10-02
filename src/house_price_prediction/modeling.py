from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, MODEL_PATH, NUMERIC_FEATURES, TARGET_COLUMN


def build_preprocessor() -> ColumnTransformer:
    """Creates a robust preprocessing pipeline for numeric and categorical inputs."""
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_model_registry() -> dict:
    """Define all benchmark models required by the syllabus and evaluation criteria."""
    return {
        "Simple Linear Regression": Pipeline(
            steps=[
                ("preprocessor", ColumnTransformer([("num", StandardScaler(), ["area"])], remainder="drop")),
                ("model", LinearRegression()),
            ]
        ),
        "Multiple Linear Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Lasso Regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", Lasso(alpha=0.001, max_iter=10000)),
            ]
        ),
        "Random Forest Regressor": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", RandomForestRegressor(
                    n_estimators=150,
                    max_depth=18,
                    min_samples_split=4,
                    random_state=42,
                )),
            ]
        ),
    }


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.DataFrame, y_test_log: pd.Series | np.ndarray) -> dict:
    """Evaluate log-transformed models on the original price scale in Crores."""
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test_log)

    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def train_and_evaluate(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Train all benchmark models and return the best pipeline plus a benchmark summary DataFrame."""
    X = df[FEATURE_COLUMNS]
    y = np.log(df[TARGET_COLUMN].astype(float))

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    benchmark_results = {}
    trained_models = {}

    for model_name, pipeline in build_model_registry().items():
        pipeline.fit(X_train, y_train_log)
        metrics = evaluate_pipeline(pipeline, X_test, y_test_log)
        benchmark_results[model_name] = metrics
        trained_models[model_name] = pipeline

    results_df = pd.DataFrame.from_dict(benchmark_results, orient="index").reset_index()
    results_df = results_df.rename(columns={"index": "Model"})
    results_df = results_df.sort_values("MAE", ascending=True).reset_index(drop=True)

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_models[best_name]
    best_pipeline_name = best_name

    diagnostics = {
        "actual_prices": np.exp(y_test_log).tolist(),
        "predicted_prices": np.exp(best_pipeline.predict(X_test)).tolist(),
        "target_log_values": y_test_log.tolist(),
        "target_values": np.exp(y_test_log).tolist(),
    }

    rf_pipeline = trained_models.get("Random Forest Regressor")
    if rf_pipeline is not None:
        preprocessor = rf_pipeline.named_steps["preprocessor"]
        rf_feature_names = preprocessor.get_feature_names_out()
        feature_importance = rf_pipeline.named_steps["model"].feature_importances_
        diagnostics["feature_importance"] = {
            "feature_names": rf_feature_names.tolist(),
            "importance": feature_importance.tolist(),
        }

    return {
        "best_pipeline": best_pipeline,
        "best_model_name": best_pipeline_name,
        "trained_models": trained_models,
        "diagnostics": diagnostics,
        "metrics": benchmark_results,
    }, results_df


def save_model_bundle(bundle: dict) -> None:
    """Persist the best pipeline and the random-forest reference model together."""
    model_bundle = {
        "best_pipeline": bundle["best_pipeline"],
        "random_forest_pipeline": bundle["trained_models"].get("Random Forest Regressor"),
        "model_name": bundle["best_model_name"],
        "diagnostics": bundle["diagnostics"],
        "metrics": bundle["metrics"],
    }
    import joblib
    joblib.dump(model_bundle, MODEL_PATH)


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    import joblib
    return joblib.load(path)
