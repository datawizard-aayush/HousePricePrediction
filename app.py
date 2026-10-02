from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.house_price_prediction.config import METADATA_PATH, MODEL_PATH
from src.house_price_prediction.modeling import load_model_bundle


@st.cache_resource
def load_project_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model is missing. Please run python train.py first.")

    if not METADATA_PATH.exists():
        raise FileNotFoundError("The metadata file is missing. Please run python train.py first.")

    with METADATA_PATH.open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    bundle = load_model_bundle(MODEL_PATH)
    return bundle, metadata


bundle, metadata = load_project_assets()

st.set_page_config(page_title="Mumbai House Price Predictor", layout="wide")
st.title("Mumbai House Price Prediction Dashboard")
st.caption("Predicting residential property prices across Mumbai using a cleaned, region-aware regression pipeline.")

region_options = metadata.get("regions", [])
type_options = metadata.get("property_types", ["Apartment", "Studio Apartment", "Villa", "Independent House", "Penthouse"])
status_options = metadata.get("statuses", ["Ready to move", "Under Construction"])
age_options = metadata.get("age_categories", ["New", "Resale", "Unknown"])

status = st.selectbox("Construction Status", options=status_options)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox("Region", options=region_options, index=0 if region_options else 0)
        bhk = st.radio("BHK", options=list(range(1, 7)), horizontal=True, index=1)
        area = st.number_input("Carpet Area (sq ft)", min_value=150, max_value=8000, value=850, step=10)
        property_type = st.selectbox("Property Type", options=type_options)

    with col2:
        if status == "Ready to move":
            age_category = st.selectbox("Building Age Category", options=age_options)
        else:
            age_category = "New"
            st.caption("Building age is set to New for under-construction properties.")

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    input_df = pd.DataFrame([
        {
            "region": region,
            "bhk": bhk,
            "type": property_type,
            "area": area,
            "status": status,
            "age": age_category,
        }
    ])

    model = bundle["best_pipeline"]
    pred_log = model.predict(input_df)[0]
    predicted_crores = float(np.exp(pred_log))
    rate_per_sqft = (predicted_crores * 10000000.0) / area
    model_mae = bundle["metrics"][bundle["model_name"]]["MAE"]

    st.success(f"Estimated Price: ₹{predicted_crores:,.2f} Cr")
    st.write(f"Holdout MAE: ₹{model_mae:,.2f} Cr")
    st.write(f"Approx. rate: ₹{rate_per_sqft:,.0f} per sq ft")

    with st.expander("Model details"):
        st.write("Best model:", bundle["model_name"])
        st.write("All prices are expressed in Crores.")

    diagnostics = bundle["diagnostics"]
    actual = np.array(diagnostics["actual_prices"])
    predicted = np.array(diagnostics["predicted_prices"])
    raw_target = np.array(diagnostics["target_values"])
    log_target = np.array(diagnostics["target_log_values"])

    tab1, tab2, tab3 = st.tabs(["Diagnostics", "Price Distribution", "Feature Importance"])

    with tab1:
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        ax1.scatter(actual, predicted, alpha=0.5, s=20)
        ax1.plot([actual.min(), actual.max()], [actual.min(), actual.max()], color="red", linestyle="--")
        ax1.set_xlabel("Actual Price (Crores)")
        ax1.set_ylabel("Predicted Price (Crores)")
        ax1.set_title("Actual vs Predicted Prices")
        st.pyplot(fig1)

    with tab2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(raw_target, bins=40, alpha=0.7, label="Price", color="steelblue")
        ax2.hist(np.exp(log_target), bins=40, alpha=0.5, label="Log-Transformed Base", color="orange")
        ax2.set_title("Target Distribution (Raw and Log-Transformed)")
        ax2.set_xlabel("Price (Crores)")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)

    with tab3:
        rf_pipeline = bundle.get("random_forest_pipeline")
        if rf_pipeline is not None:
            preprocessor = rf_pipeline.named_steps["preprocessor"]
            feature_names = preprocessor.get_feature_names_out()
            importances = rf_pipeline.named_steps["model"].feature_importances_
            importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            importance_df = importance_df.sort_values("importance", ascending=False).head(10)

            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.barh(importance_df["feature"][::-1], importance_df["importance"][::-1])
            ax3.set_title("Top Random Forest Feature Importance")
            ax3.set_xlabel("Importance")
            st.pyplot(fig3)
        else:
            st.info("Random Forest model is not available for feature importance plotting.")

else:
    st.info("Fill in the property details and click Estimate Price to generate the prediction.")
