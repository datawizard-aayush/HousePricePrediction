"""Prediction intervals and shared evaluation artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .modeling import RANDOM_STATE, _pipeline


def train_prediction_interval_models(
    X_train: pd.DataFrame, y_train_log: pd.Series
) -> dict:
    """Fit 10th/90th conditional quantile models on log prices (Unit 5 uncertainty)."""
    models = {}
    for quantile in (0.1, 0.9):
        print(f"  Fitting {quantile:.0%} price quantile model...", flush=True)
        estimator = _pipeline(
            HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                max_iter=180,
                random_state=RANDOM_STATE,
            )
        )
        estimator.fit(X_train, y_train_log)
        models[f"q{int(quantile * 100)}"] = estimator
    return models


def predict_price_range(models: dict, features: pd.DataFrame) -> tuple[float, float]:
    """Return an ordered 10th-90th interval in Crores, containing the point estimate."""
    low, high = (float(np.exp(models[key].predict(features)[0])) for key in ("q10", "q90"))
    return min(low, high), max(low, high)
