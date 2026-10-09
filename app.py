"""Streamlit dashboard for model predictions and precomputed course diagnostics."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.house_price_prediction.config import (
    DIAGNOSTICS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.house_price_prediction.evaluation import predict_price_range
from src.house_price_prediction.modeling import load_model_bundle


st.set_page_config(page_title="Mumbai House Price Prediction", layout="wide")


@st.cache_resource
def load_project_assets() -> tuple[dict, dict, dict]:
    """Load trained assets only; fitting models always happens in train.py."""
    required = (MODEL_PATH, METADATA_PATH, DIAGNOSTICS_PATH)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Training artifacts are missing: "
            + ", ".join(missing)
            + ". Run `python train.py` from the project root, then reload this page."
        )
    with METADATA_PATH.open("r", encoding="utf-8") as source:
        metadata = json.load(source)
    with DIAGNOSTICS_PATH.open("r", encoding="utf-8") as source:
        diagnostics = json.load(source)
    return load_model_bundle(MODEL_PATH), metadata, diagnostics


try:
    bundle, metadata, diagnostics = load_project_assets()
except (FileNotFoundError, OSError, ValueError) as error:
    st.error(f"Dashboard assets could not be loaded. {error}")
    st.stop()

st.title("Mumbai House Price Prediction Dashboard")
st.caption(
    "Modeling asking prices in Crores. Cross-fitted target encoding preserves high-cardinality location signals."
)


def show_figure(name: str, caption: str | None = None) -> None:
    path = REPORTS_DIR / name
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure {name} is not available. Run `python train.py` to regenerate artifacts.")


def show_image_matrix(matrix: list[list[int]], labels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title=title,
        xlabel="Predicted tier",
        ylabel="Actual tier",
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)


def numeric_sensitivity(model, features: pd.DataFrame) -> pd.DataFrame:
    """Show local numeric-feature changes around the current prediction (Unit 6 XAI)."""
    result = []
    point = float(np.exp(model.predict(features)[0]))
    for feature in ("area", "bhk"):
        changed = features.copy()
        if feature == "area":
            changed.loc[:, feature] = max(150, float(features.iloc[0][feature]) * 0.9)
        else:
            changed.loc[:, feature] = max(1, float(features.iloc[0][feature]) - 1)
        changed.loc[:, "area_per_bhk"] = changed["area"] / changed["bhk"]
        alternate = float(np.exp(model.predict(changed)[0]))
        result.append(
            {
                "Changed input": feature,
                "Comparison": "10% less area" if feature == "area" else "One fewer BHK",
                "Estimate change (Cr)": alternate - point,
            }
        )
    return pd.DataFrame(result)


status_options = metadata.get("statuses", ["Ready to move", "Under Construction"])
regions = metadata.get("regions", [])
property_types = metadata.get("property_types", ["Apartment"])
age_options = metadata.get("age_categories", ["New", "Resale", "Unknown"])

with st.sidebar:
    st.header("Property details")
    status = st.selectbox("Construction Status", status_options)
    region = st.selectbox("Region", regions)

locality_options = metadata.get("region_to_localities", {}).get(region, [])
locality_options = sorted(set(locality_options + ["Other"]))
if not locality_options:
    locality_options = ["Other"]

with st.form("prediction_form"):
    left, right = st.columns(2)
    with left:
        locality = st.selectbox("Locality", locality_options)
        bhk = st.selectbox("BHK", range(1, 7), index=1)
        area = st.number_input(
            "Carpet Area (sq ft)", min_value=150, max_value=8000, value=850, step=10
        )
    with right:
        property_type = st.selectbox("Property Type", property_types)
        if status == "Ready to move":
            age_category = st.selectbox("Building Age Category", age_options)
        else:
            age_category = "New"
            st.caption("Building age is set to New for under-construction properties.")
        include_asking_price = st.checkbox("Compare with an asking price")
        asking_price = (
            st.number_input("Asking price (Cr)", min_value=0.01, value=1.0, step=0.05)
            if include_asking_price
            else None
        )
    submitted = st.form_submit_button("Estimate Price", type="primary")

if submitted:
    input_row = {
        "region": region,
        "locality": locality,
        "bhk": int(bhk),
        "type": property_type,
        "area": float(area),
        "area_per_bhk": float(area) / int(bhk),
        "status": status,
        "age": age_category,
    }
    input_frame = pd.DataFrame([input_row])
    point = float(np.exp(bundle["best_pipeline"].predict(input_frame)[0]))
    interval_low, interval_high = predict_price_range(
        bundle["prediction_interval_models"], input_frame
    )
    st.session_state["last_prediction"] = {
        "features": input_frame,
        "point": point,
        "low": min(point, interval_low, interval_high),
        "high": max(point, interval_low, interval_high),
        "asking_price": asking_price,
    }

tabs = st.tabs(
    [
        "Prediction",
        "Similar Listings",
        "Model Comparison",
        "Diagnostics",
    ]
)

with tabs[0]:
    st.subheader("Prediction")
    prediction_state = st.session_state.get("last_prediction")
    if prediction_state is None:
        st.info("Enter property details and submit the form to estimate a price.")
    else:
        point = prediction_state["point"]
        low, high = prediction_state["low"], prediction_state["high"]
        frame = prediction_state["features"]
        rate_per_sqft = point * 10_000_000 / float(frame.iloc[0]["area"])
        region_median_rate = metadata.get("region_median_price_per_sqft", {}).get(region)
        a, b, c = st.columns(3)
        a.metric("Estimated price", f"₹{point:,.2f} Cr")
        b.metric("10th–90th percentile range", f"₹{low:,.2f}–{high:,.2f} Cr")
        c.metric("Estimated rate", f"₹{rate_per_sqft:,.0f}/sq ft")
        if region_median_rate:
            st.write(
                f"Region median: ₹{region_median_rate:,.0f}/sq ft "
                f"({(rate_per_sqft / region_median_rate - 1) * 100:+.1f}% vs. median)."
            )
        classifier = bundle["best_classifier"]
        class_index = int(classifier.predict(frame)[0])
        probabilities = classifier.predict_proba(frame)[0]
        tier_names = bundle["price_tier_names"]
        st.write(
            f"**Predicted price tier:** {tier_names[class_index]} "
            f"({bundle['best_classifier_name']})"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Price tier": tier_names,
                    "Probability": probabilities,
                }
            ).assign(Probability=lambda table: table["Probability"].map(lambda p: f"{p:.1%}")),
            hide_index=True,
            use_container_width=True,
        )
        asking = prediction_state["asking_price"]
        if asking is not None:
            difference = (asking / point - 1) * 100
            verdict = "Overpriced" if difference > 10 else "Underpriced" if difference < -10 else "Fair deal"
            st.info(f"**{verdict}** — asking price is {difference:+.1f}% vs. the estimate.")
            st.caption("Verdicts use a ±10% comparison band; they are not financial advice.")
        st.caption(
            f"Best regression model: {bundle['best_model_name']} · "
            f"holdout MAE: ₹{bundle['metrics'][bundle['best_model_name']]['MAE']:.2f} Cr."
        )

with tabs[1]:
    st.subheader("Similar Listings")
    prediction_state = st.session_state.get("last_prediction")
    listings = bundle.get("similar_listings")
    if prediction_state is None:
        st.info("Submit a prediction to find comparable real listings.")
    elif listings is None or listings.empty:
        st.warning("Similar-listing data was not saved. Retrain with the current train.py.")
    else:
        query = prediction_state["features"].iloc[0]
        candidates = listings[listings["region"] == query["region"]].copy()
        if candidates.empty:
            candidates = listings.copy()
        candidates["distance"] = (
            np.abs(np.log(candidates["area"].clip(lower=1) / query["area"]))
            + 0.7 * np.abs(candidates["bhk"] - query["bhk"])
            + 0.6 * (candidates["type"] != query["type"]).astype(float)
            + 0.4 * (candidates["locality"] != query["locality"]).astype(float)
        )
        shown = candidates.nsmallest(5, "distance")
        columns = ["region", "locality", "bhk", "type", "area", "status", TARGET_COLUMN, "price_per_sqft"]
        st.dataframe(shown[columns].rename(columns={TARGET_COLUMN: "price_cr"}), hide_index=True, use_container_width=True)
        st.caption("Ranked within the selected region by area, BHK, property type, and locality.")

with tabs[2]:
    st.subheader("Model Comparison")
    regression_metrics = pd.DataFrame.from_dict(bundle["metrics"], orient="index").rename_axis("Model")
    st.dataframe(regression_metrics.style.format(precision=3), use_container_width=True)
    show_figure("regression_model_comparison.png", "5-fold CV MAE with standard deviation.")
    classifier_metrics = pd.DataFrame.from_dict(
        diagnostics["classification_metrics"], orient="index"
    ).drop(columns=["Confusion_Matrix"])
    st.markdown("#### Price-tier classification")
    st.dataframe(classifier_metrics.style.format(precision=3), use_container_width=True)
    class_names = diagnostics["price_tier_names"]
    best_classification = bundle["classification_metrics"][bundle["best_classifier_name"]]
    show_image_matrix(
        best_classification["Confusion_Matrix"],
        class_names,
        f"{bundle['best_classifier_name']}: Confusion Matrix",
    )
    roc_data = diagnostics["classification_curves"][bundle["best_classifier_name"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    for tier, curve in roc_data.items():
        ax.plot(curve["fpr"], curve["tpr"], label=tier)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="One-vs-Rest ROC Curves")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

with tabs[3]:
    st.subheader("Diagnostics")
    regression = diagnostics["regression"]
    actual = np.asarray(regression["actual_prices"])
    predicted = np.asarray(regression["predicted_prices"])
    residuals = np.asarray(regression["residuals"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(actual, predicted, alpha=0.45, s=12)
    axes[0].plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--")
    axes[0].set(title="Actual vs Predicted", xlabel="Actual (Cr)", ylabel="Predicted (Cr)")
    axes[1].scatter(predicted, residuals, alpha=0.45, s=12)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set(title="Residual Plot", xlabel="Predicted (Cr)", ylabel="Residual (Cr)")
    st.pyplot(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist(residuals, bins=45, color="slateblue", alpha=0.8)
    axes[0].set(title="Residual Histogram", xlabel="Error (Cr)", ylabel="Frequency")
    axes[1].hist(diagnostics["price_distribution"]["raw"], bins=45, color="steelblue", alpha=0.75)
    axes[1].set(title="Raw Price Distribution", xlabel="Price (Cr)", ylabel="Listings")
    axes[2].hist(diagnostics["price_distribution"]["log"], bins=45, color="darkorange", alpha=0.75)
    axes[2].set(title="Log Price Distribution", xlabel="Log(Price in Cr)", ylabel="Listings")
    st.pyplot(fig)
    plt.close(fig)

