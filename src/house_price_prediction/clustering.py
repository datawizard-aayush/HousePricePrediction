"""Region-level market segmentation with clustering and PCA (syllabus Unit 4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import REPORTS_DIR, TARGET_COLUMN

RANDOM_STATE = 42
MARKET_TIER_NAMES = [
    "Affordable outskirts",
    "Mid-market",
    "Premium suburbs",
    "Luxury core",
]


def _market_name(rank: int, count: int) -> str:
    level = min(3, int(rank * 4 / count))
    return MARKET_TIER_NAMES[level]


def build_market_segments(training_data: pd.DataFrame, figures_dir: Path = REPORTS_DIR) -> dict:
    """Cluster training-only region aggregates; save elbow, silhouette, PCA, and dendrogram figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    region_table = (
        training_data.groupby("region", observed=True)
        .agg(
            median_price_per_sqft=("price_per_sqft", "median"),
            median_price_cr=(TARGET_COLUMN, "median"),
            listing_count=(TARGET_COLUMN, "size"),
            median_area=("area", "median"),
        )
        .reset_index()
    )
    if len(region_table) < 3:
        raise ValueError("At least three regions are required for market segmentation.")
    cluster_features = region_table.drop(columns="region")
    scaled = StandardScaler().fit_transform(cluster_features)
    k_values = list(range(2, min(10, len(region_table) - 1) + 1))
    inertias, silhouettes = [], []
    models = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(scaled)
        models[k] = (model, labels)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(scaled, labels)))
    chosen_k = k_values[int(np.argmax(silhouettes))]
    cluster_model, labels = models[chosen_k]
    mixture_results = []
    for k in k_values:
        mixture = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_STATE)
        mixture_labels = mixture.fit_predict(scaled)
        mixture_results.append(
            {
                "k": k,
                "bic": float(mixture.bic(scaled)),
                "aic": float(mixture.aic(scaled)),
                "silhouette": float(silhouette_score(scaled, mixture_labels)),
            }
        )

    # Unit 4: give numeric clusters market-facing names ordered by median price/sq ft.
    cluster_prices = region_table.assign(cluster=labels).groupby("cluster")[
        "median_price_per_sqft"
    ].median().sort_values()
    cluster_names = {
        int(cluster): _market_name(rank, len(cluster_prices))
        for rank, cluster in enumerate(cluster_prices.index)
    }
    region_table["cluster_id"] = labels
    region_table["cluster_name"] = [cluster_names[int(label)] for label in labels]
    region_table["cluster"] = region_table["cluster_name"]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(scaled)
    region_table["pca_1"], region_table["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    mapping = dict(zip(region_table["region"], region_table["cluster_name"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(k_values, inertias, marker="o")
    axes[0].set(title="K-Means Elbow Curve", xlabel="Number of clusters (k)", ylabel="Inertia")
    axes[1].plot(k_values, silhouettes, marker="o", color="darkorange")
    axes[1].axvline(chosen_k, linestyle="--", color="grey")
    axes[1].set(title="Silhouette Score by k", xlabel="Number of clusters (k)", ylabel="Silhouette")
    fig.tight_layout()
    fig.savefig(figures_dir / "market_elbow_silhouette.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([row["k"] for row in mixture_results], [row["bic"] for row in mixture_results], marker="o", label="BIC")
    ax.plot([row["k"] for row in mixture_results], [row["aic"] for row in mixture_results], marker="o", label="AIC")
    ax.set(title="Gaussian Mixture Model Comparison", xlabel="Number of components", ylabel="Information criterion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "market_gmm_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, group in region_table.groupby("cluster_name", observed=True):
        ax.scatter(group["pca_1"], group["pca_2"], label=name, alpha=0.8)
    ax.set(title="Mumbai Region Clusters (PCA)", xlabel="PC1", ylabel="PC2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "market_pca_clusters.png", dpi=150)
    plt.close(fig)

    hierarchy = linkage(scaled, method="ward")
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(hierarchy, labels=region_table["region"].tolist(), leaf_rotation=90, leaf_font_size=6, ax=ax)
    ax.set_title("Hierarchical Region Clustering (Ward)")
    fig.tight_layout()
    fig.savefig(figures_dir / "market_dendrogram.png", dpi=150)
    plt.close(fig)

    return {
        "region_clusters": mapping,
        "region_summary": region_table.to_dict(orient="records"),
        "cluster_diagnostics": {
            "k_values": k_values,
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "selected_k": chosen_k,
            "gaussian_mixture": mixture_results,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "region_features": cluster_features.columns.tolist(),
        },
        "cluster_model": cluster_model,
        "cluster_scaler": StandardScaler().fit(cluster_features),
    }


def add_cluster_fairness_tier(data: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    """Map each region to its precomputed market cluster for fairness-style reporting."""
    return data["region"].map(mapping).fillna("Unmapped")
