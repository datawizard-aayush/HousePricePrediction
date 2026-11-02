"""Price-tier classification models and holdout metrics (syllabus Units 3 and 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .modeling import RANDOM_STATE, _pipeline

PRICE_TIER_NAMES = ["Budget", "Mid", "Premium", "Luxury"]


def train_classifiers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_cr: pd.Series,
    y_test_cr: pd.Series,
) -> dict:
    """Fit the selected Random Forest classifier and retain holdout metrics."""
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

    name = "Random Forest Classifier"
    pipeline = _pipeline(
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        target_type="multiclass",
    )
    print(f"  Fitting {name}...", flush=True)
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test).astype(int)
    probabilities = pipeline.predict_proba(X_test)
    auc = float(
        roc_auc_score(y_test, probabilities, labels=class_ids, multi_class="ovr", average="macro")
    )
    results = {
        name: {
            "Accuracy": float(accuracy_score(y_test, predicted)),
            "Precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
            "Recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
            "F1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "ROC_AUC_ovr_macro": auc,
            "Confusion_Matrix": confusion_matrix(y_test, predicted, labels=class_ids).tolist(),
        }
    }
    curves = {
        name: {
            tier: dict(zip(
                ("fpr", "tpr"),
                (curve.tolist() for curve in roc_curve(y_test == class_id, probabilities[:, class_id])[:2]),
            ))
            for class_id, tier in enumerate(names)
        }
    }
    best_name = name
    return {
        "best_classifier": pipeline,
        "best_classifier_name": best_name,
        "classifiers": {name: pipeline},
        "classification_metrics": results,
        "classification_curves": curves,
        "price_tier_names": names,
        "price_tier_edges": edges.tolist(),
        "classification_holdout": {
            "actual": y_test.tolist(),
            "predicted": predicted.tolist(),
            "probabilities": probabilities.tolist(),
        },
        "classification_test_labels": y_test.tolist(),
    }


def one_vs_rest_confusion_matrix(
    confusion_counts: list[list[int]], positive_index: int
) -> list[list[int]]:
    """Convert a multiclass matrix into [[TN, FP], [FN, TP]] for one class."""
    true_positive = confusion_counts[positive_index][positive_index]
    false_negative = sum(confusion_counts[positive_index]) - true_positive
    false_positive = sum(row[positive_index] for row in confusion_counts) - true_positive
    total = sum(sum(row) for row in confusion_counts)
    true_negative = total - true_positive - false_negative - false_positive
    return [[true_negative, false_positive], [false_negative, true_positive]]


def save_classification_figures(bundle: dict, figures_dir) -> None:
    """Save a one-vs-rest confusion matrix and ROC curves for the selected classifier."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = bundle["price_tier_names"]
    selected = bundle["best_classifier_name"]
    positive_index = names.index("Luxury") if "Luxury" in names else len(names) - 1
    matrix = one_vs_rest_confusion_matrix(
        bundle["classification_metrics"][selected]["Confusion_Matrix"], positive_index
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title=f"{selected}: One-vs-Rest ({names[positive_index]})",
        xlabel="Predicted",
        ylabel="Actual",
        xticks=range(2),
        yticks=range(2),
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{cell_names[row][column]}: {matrix[row][column]}",
                ha="center",
                va="center",
            )
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
