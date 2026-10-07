"""Price-tier classification models and holdout metrics (syllabus Units 3 and 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier

from .config import TARGET_COLUMN
from .modeling import RANDOM_STATE, _pipeline

PRICE_TIER_NAMES = ["Budget", "Mid", "Premium", "Luxury"]


def train_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_cr: pd.Series,
    y_test_cr: pd.Series,
) -> dict:
    """Fit three classifiers with train-only quantile thresholds and retain holdout metrics."""
    # Unit 3: classify price tiers, then report confusion matrix and macro/OVR metrics.
    _, edges = pd.qcut(
        y_train_cr,
        q=4,
        labels=False,
        retbins=True,
        duplicates="drop",
    )
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Training prices do not contain enough distinct values for price tiers.")
    names = PRICE_TIER_NAMES[: len(edges) - 1]
    bins = edges.copy()
    bins[0], bins[-1] = -np.inf, np.inf
    y_train = pd.cut(y_train_cr, bins, labels=False, include_lowest=True)
    y_test = pd.cut(y_test_cr, bins, labels=False, include_lowest=True)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    class_ids = np.arange(len(names))

    candidates = {
        "Logistic Regression": _pipeline(
            LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
            target_type="multiclass",
        ),
        "KNN Classifier": _pipeline(
            KNeighborsClassifier(n_neighbors=15, n_jobs=-1), target_type="multiclass"
        ),
        "Random Forest Classifier": _pipeline(
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            target_type="multiclass",
        ),
    }
    results, fitted, curves = {}, {}, {}
    for name, pipeline in candidates.items():
        print(f"  Fitting {name}...", flush=True)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test).astype(int)
        probabilities = pipeline.predict_proba(X_test)
        auc = float(
            roc_auc_score(y_test, probabilities, labels=class_ids, multi_class="ovr", average="macro")
        )
        results[name] = {
            "Accuracy": float(accuracy_score(y_test, predicted)),
            "Precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
            "Recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
            "F1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "ROC_AUC_ovr_macro": auc,
            "Confusion_Matrix": confusion_matrix(y_test, predicted, labels=class_ids).tolist(),
        }
        curves[name] = {
            tier: dict(zip(
                ("fpr", "tpr"),
                (curve.tolist() for curve in roc_curve(y_test == class_id, probabilities[:, class_id])[:2]),
            ))
            for class_id, tier in enumerate(names)
        }
        fitted[name] = pipeline
    best_name = max(results, key=lambda name: results[name]["F1_macro"])
    return {
        "best_classifier": fitted[best_name],
        "best_classifier_name": best_name,
        "classifiers": fitted,
        "classification_metrics": results,
        "classification_curves": curves,
        "price_tier_names": names,
        "price_tier_edges": edges.tolist(),
        "classification_holdout": {
            "actual": y_test.tolist(),
            "predicted": fitted[best_name].predict(X_test).astype(int).tolist(),
            "probabilities": fitted[best_name].predict_proba(X_test).tolist(),
        },
        "classification_test_labels": y_test.tolist(),
    }


def save_classification_figures(bundle: dict, figures_dir) -> None:
    """Save the selected classifier's confusion matrix and one-vs-rest ROC curves."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = bundle["price_tier_names"]
    selected = bundle["best_classifier_name"]
    matrix = bundle["classification_metrics"][selected]["Confusion_Matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title=f"{selected}: Confusion Matrix",
        xlabel="Predicted tier",
        ylabel="Actual tier",
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
    )
    for row in range(len(names)):
        for column in range(len(names)):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for tier, curve in bundle["classification_curves"][selected].items():
        ax.plot(curve["fpr"], curve["tpr"], label=tier)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(title=f"{selected}: One-vs-Rest ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "classification_roc_curves.png", dpi=150)
    plt.close(fig)
