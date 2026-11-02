"""Selected regression pipeline and leakage-safe tuning routines."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.base import clone

from .config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    HIGH_CARDINALITY_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

RANDOM_STATE = 42


def build_preprocessor(target_type: str = "auto") -> ColumnTransformer:
    """Build the shared encoder; TargetEncoder cross-fitting prevents in-fold target leakage."""
    # Unit 5: scaling is required for distance- and margin-based estimators.
    encoder_cv = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        if target_type in ("binary", "multiclass")
        else KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "high_cardinality",
                TargetEncoder(
                    smooth="auto",
                    cv=encoder_cv,
                    target_type=target_type,
                ),
                HIGH_CARDINALITY_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _pipeline(
    estimator, features: list[str] | None = None, target_type: str = "auto"
) -> Pipeline:
    preprocessor = build_preprocessor(target_type)
    if features is not None:
        preprocessor = ColumnTransformer(
            [("numeric", StandardScaler(), features)],
            remainder="drop",
        )
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def build_model_registry() -> dict[str, Pipeline]:
    """Build the selected Random Forest price-regression pipeline."""
    return {
        "Random Forest Regressor": _pipeline(
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        ),
    }


def evaluate_pipeline(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test_log: pd.Series | np.ndarray
) -> dict[str, float]:
    """Evaluate a log-price model on the original Crores scale."""
    prediction = np.exp(pipeline.predict(X_test))
    actual = np.exp(np.asarray(y_test_log))
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
        "R2": float(r2_score(actual, prediction)),
    }


def _negative_crore_mae(estimator, features, target_log) -> float:
    return -mean_absolute_error(np.exp(target_log), np.exp(estimator.predict(features)))


def _tune_estimators(
    models: dict[str, Pipeline], X_train: pd.DataFrame, y_train: pd.Series
) -> dict[str, dict]:
    """Tune the selected Random Forest using only the training partition."""
    tuning_specs = {
        "Random Forest Regressor": {
            "model__n_estimators": [100, 150, 200],
            "model__max_depth": [None, 18, 28],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [0.7, 1.0],
        }
    }
    best_params = {}
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for name, parameters in tuning_specs.items():
        print(f"  Tuning {name} (randomized search, CV=3)...", flush=True)
        search = RandomizedSearchCV(
            estimator=models[name],
            param_distributions=parameters,
            scoring=_negative_crore_mae,
            cv=inner_cv,
            n_iter=15,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train)
        models[name] = search.best_estimator_
        best_params[name] = search.best_params_
    return best_params


def train_regression(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Tune and evaluate the selected Random Forest regressor."""
    features = df[FEATURE_COLUMNS]
    target_log = np.log(df[TARGET_COLUMN].astype(float))
    X_train, X_test, y_train, y_test = train_test_split(
        features, target_log, test_size=0.2, random_state=RANDOM_STATE
    )
    models = build_model_registry()
    print("Tuning the Random Forest regressor on the 80% training partition...", flush=True)
    best_params = _tune_estimators(models, X_train, y_train)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    name, candidate = next(iter(models.items()))
    print(f"  5-fold CV: {name}...", flush=True)
    mae_scores, r2_scores = [], []
    for train_indices, validation_indices in cv.split(X_train):
        fold_model = clone(candidate)
        fold_X = X_train.iloc[train_indices]
        fold_y = y_train.iloc[train_indices]
        fold_model.fit(fold_X, fold_y)
        validation_X = X_train.iloc[validation_indices]
        validation_actual = np.exp(y_train.iloc[validation_indices])
        validation_predicted = np.exp(fold_model.predict(validation_X))
        mae_scores.append(mean_absolute_error(validation_actual, validation_predicted))
        r2_scores.append(r2_score(validation_actual, validation_predicted))

    best_pipeline = clone(candidate)
    best_pipeline.fit(X_train, y_train)
    fitted_models = {name: best_pipeline}
    holdout = evaluate_pipeline(best_pipeline, X_test, y_test)
    results.append(
        {
            "Model": name,
            "CV_MAE_Mean": float(np.mean(mae_scores)),
            "CV_MAE_Std": float(np.std(mae_scores)),
            "CV_R2_Mean": float(np.mean(r2_scores)),
            "CV_R2_Std": float(np.std(r2_scores)),
            "Test_MAE": holdout["MAE"],
            "Test_RMSE": holdout["RMSE"],
            "Test_R2": holdout["R2"],
        }
    )
    results_df = pd.DataFrame(results)
    best_name = name
    predicted_test = np.exp(best_pipeline.predict(X_test))
    actual_test = np.exp(y_test)
    diagnostics = {
        "actual_prices": actual_test.tolist(),
        "predicted_prices": predicted_test.tolist(),
        "residuals": (actual_test - predicted_test).tolist(),
        "target_log_values": y_test.tolist(),
        "target_values": actual_test.tolist(),
    }
    bundle = {
        "best_pipeline": best_pipeline,
        "best_model_name": best_name,
        "trained_models": fitted_models,
        "diagnostics": diagnostics,
        "metrics": {
            row["Model"]: {
                "CV_MAE_Mean": row["CV_MAE_Mean"],
                "CV_MAE_Std": row["CV_MAE_Std"],
                "CV_R2_Mean": row["CV_R2_Mean"],
                "CV_R2_Std": row["CV_R2_Std"],
                "MAE": row["Test_MAE"],
                "RMSE": row["Test_RMSE"],
                "R2": row["Test_R2"],
            }
            for row in results
        },
        "best_params": best_params,
        "test_features": X_test.reset_index(drop=True),
        "test_target_log": y_test.reset_index(drop=True),
        "similar_listings": df.loc[X_train.index].reset_index(drop=True),
    }
    return bundle, results_df, X_train, X_test, y_train


def save_model_bundle(bundle: dict, path: str | Path = MODEL_PATH) -> None:
    """Persist the dashboard-ready model bundle."""
    model_bundle = dict(bundle)
    model_bundle["model_name"] = model_bundle["best_model_name"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, path)


def load_model_bundle(path: str | Path = MODEL_PATH) -> dict:
    """Load either the current bundle or the legacy pipeline bundle."""
    return joblib.load(path)
