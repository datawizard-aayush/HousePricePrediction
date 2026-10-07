"""Holdout explainability, learning/validation curves, and group error checks."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeRegressor

from .config import REPORTS_DIR
from .modeling import RANDOM_STATE, _negative_crore_mae, _pipeline


def build_explainability(
    best_model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test_log: pd.Series,
    figures_dir: Path = REPORTS_DIR,
) -> dict:
    """Generate best-model permutation importance and Unit 5 learning diagnostics."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    print("  Calculating holdout permutation importance...", flush=True)
    importance = permutation_importance(
        best_model,
        X_test,
        y_test_log,
        scoring=_negative_crore_mae,
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance_results = [
        {"feature": name, "importance_mean": float(mean), "importance_std": float(std)}
        for name, mean, std in zip(X_test.columns, importance.importances_mean, importance.importances_std)
    ]
    importance_results.sort(key=lambda item: item["importance_mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    top = importance_results[: min(10, len(importance_results))]
    ax.barh(
        [item["feature"] for item in top][::-1],
        [item["importance_mean"] for item in top][::-1],
        xerr=[item["importance_std"] for item in top][::-1],
    )
    ax.set(title="Permutation Importance (holdout, Crore MAE)", xlabel="MAE reduction (Cr)")
    fig.tight_layout()
    fig.savefig(figures_dir / "permutation_importance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    partial_dependence_data = X_test.copy()
    partial_dependence_data["area"] = partial_dependence_data["area"].astype(float)
    partial_dependence_data["bhk"] = partial_dependence_data["bhk"].astype(float)
    PartialDependenceDisplay.from_estimator(
        best_model,
        partial_dependence_data,
        ["area", "bhk"],
        kind="average",
        ax=axes,
        grid_resolution=20,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "partial_dependence.png", dpi=150)
    plt.close(fig)

    return {
        "permutation_importance": importance_results,
        "figures": {
            "permutation_importance": "permutation_importance.png",
            "partial_dependence": "partial_dependence.png",
        },
    }


def build_learning_diagnostics(
    best_model,
    X_train: pd.DataFrame,
    y_train_log: pd.Series,
    figures_dir: Path = REPORTS_DIR,
) -> dict:
    """Compare shallow/deep decision trees with the CV-best model."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(X_train), 12000)
    selected = X_train.sample(n=n, random_state=RANDOM_STATE).index
    X = X_train.loc[selected]
    y = y_train_log.loc[selected]
    curves = {}
    candidates = {
        "Decision Tree (low depth)": _pipeline(
            DecisionTreeRegressor(max_depth=3, random_state=RANDOM_STATE)
        ),
        "Decision Tree (high depth)": _pipeline(
            DecisionTreeRegressor(max_depth=24, random_state=RANDOM_STATE)
        ),
        "Best model": clone(best_model),
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, estimator in candidates.items():
        sizes, train_scores, validation_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=[0.2, 0.5, 1.0],
            cv=3,
            scoring=_negative_crore_mae,
            shuffle=True,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        train_mae = -train_scores.mean(axis=1)
        validation_mae = -validation_scores.mean(axis=1)
        curves[name] = {
            "training_sizes": sizes.tolist(),
            "train_mae": train_mae.tolist(),
            "validation_mae": validation_mae.tolist(),
        }
        ax.plot(sizes, validation_mae, marker="o", label=f"{name} (validation)")
        ax.plot(sizes, train_mae, linestyle="--", alpha=0.65, label=f"{name} (training)")
    ax.set(title="Learning Curves: Overfitting, Underfitting & Bias-Variance", xlabel="Training rows", ylabel="MAE (Cr)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "learning_curves.png", dpi=150)
    plt.close(fig)

    tree = _pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE))
    depths = [2, 3, 5, 8, 12, 18, 24, 32]
    train_scores, validation_scores = validation_curve(
        tree,
        X,
        y,
        param_name="model__max_depth",
        param_range=depths,
        cv=3,
        scoring=_negative_crore_mae,
        n_jobs=1,
    )
    validation_data = {
        "max_depth": depths,
        "train_mae": (-train_scores.mean(axis=1)).tolist(),
        "validation_mae": (-validation_scores.mean(axis=1)).tolist(),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(depths, validation_data["train_mae"], marker="o", label="Training")
    ax.plot(depths, validation_data["validation_mae"], marker="o", label="Validation")
    ax.set(title="Decision Tree Validation Curve", xlabel="max_depth", ylabel="MAE (Cr)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "decision_tree_validation_curve.png", dpi=150)
    plt.close(fig)
    return {
        "learning_curves": curves,
        "validation_curve": validation_data,
        "figures": {
            "learning_curves": "learning_curves.png",
            "validation_curve": "decision_tree_validation_curve.png",
        },
    }


def build_group_error_report(
    test_features: pd.DataFrame,
    actual_prices: list[float],
    predicted_prices: list[float],
    region_clusters: dict[str, str],
) -> dict:
    """Compare MAE and MAPE by market segment and property type."""
    data = test_features.copy().reset_index(drop=True)
    data["actual_cr"] = np.asarray(actual_prices)
    data["predicted_cr"] = np.asarray(predicted_prices)
    data["absolute_error_cr"] = (data["actual_cr"] - data["predicted_cr"]).abs()
    data["absolute_percentage_error"] = (
        data["absolute_error_cr"] / data["actual_cr"].clip(lower=1e-8) * 100
    )
    data["region_cluster"] = data["region"].map(region_clusters).fillna("Unmapped")
    reports = {}
    for group_column, key in (
        ("region_cluster", "by_region_cluster"),
        ("type", "by_property_type"),
    ):
        summary = (
            data.groupby(group_column, observed=True)
            .agg(
                Listings=("actual_cr", "size"),
                MAE_Cr=("absolute_error_cr", "mean"),
                MAPE_pct=("absolute_percentage_error", "mean"),
            )
            .reset_index()
            .rename(columns={group_column: "Group"})
            .sort_values("MAE_Cr", ascending=False)
        )
        reports[key] = summary.to_dict(orient="records")
    return reports
