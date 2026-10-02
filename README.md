# Mumbai House Price Prediction

This project implements a machine learning-based residential property valuation system for Mumbai, aligned with the provided product requirements and architecture.

## Project goals
- Predict property prices from location and structural features
- Compare linear and tree-based models
- Persist the best trained model for dashboard inference
- Expose a lightweight Streamlit UI for user-friendly predictions

## Quick start

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model:
   ```bash
   python train.py
   ```
4. Launch the UI:
   ```bash
   streamlit run app.py
   ```

## Dataset

Training uses `data/raw/mumbai_house_prices_cleaned_cr.csv`. Its `price_cr` target is modeled and displayed in Crores. The loader also accepts the legacy `price` plus `price_unit` format.

The prediction form asks for a building age category only for ready-to-move properties. Under-construction properties use the `New` category.

## Architecture summary

- `train.py`: entry point for loading data, training models, and saving artifacts
- `app.py`: Streamlit dashboard for single-property prediction
- `src/house_price_prediction/`: reusable ML logic and utilities
- `artifacts/`: saved model and evaluation metrics
- `reports/figures/`: generated diagnostic charts
