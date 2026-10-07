# Mumbai House Price Prediction

Mumbai residential listing-price prediction with scikit-learn and Streamlit. The target is `price_cr` (Crores), trained as `log(price_cr)`; evaluation metrics and displayed predictions are converted back to Crores.

## Setup

Use Python 3.12 or newer and create an isolated environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The model uses `TargetEncoder`, available in the pinned scikit-learn version. The training script also needs SciPy for Ward hierarchical clustering. SHAP is optional and is not required by the dashboard.

## Train and run

From the project root, with the cleaned dataset at `data/raw/mumbai_house_prices_cleaned_cr.csv`:

```powershell
python train.py
streamlit run app.py
```

On Windows, double-click `start.bat` to launch the dashboard. It runs `train.py` first if any required model artifacts are missing, then starts Streamlit.

Training performs a fixed 80/20 holdout, training-partition-only hyperparameter searches, 5-fold cross-validation, and writes the dashboard bundle (`model_pipeline.joblib`), selector/market metadata (`metadata.json`), precomputed diagnostics (`diagnostics.json`), benchmark CSVs under `reports/`, and figures under `reports/figures/`. The Streamlit app only loads saved artifacts; it never trains models.

After pulling code changes, run `python train.py` to regenerate compatible artifacts before launching the app. The regenerated model bundle is larger than GitHub's 100 MB per-file limit and is not included in the repository update.

The regression benchmark includes nine algorithms. SVR (RBF) uses a deterministic 10,000-row sample in each training fold and final fit to limit runtime; validation and holdout predictions still cover their complete partitions.

## Project structure

- `train.py`: training orchestration, benchmarks, persisted artifacts, and figures.
- `app.py`: prediction dashboard and precomputed analysis tabs.
- `src/house_price_prediction/config.py`: feature and artifact paths.
- `src/house_price_prediction/data_processing.py`: data cleaning and listing metadata.
- `src/house_price_prediction/modeling.py`: regression models, encoders, CV, and tuning.
- `src/house_price_prediction/classification.py`: price-tier classifiers and metrics.
- `src/house_price_prediction/clustering.py`: market segmentation, PCA, GMM, and hierarchical clustering.
- `src/house_price_prediction/evaluation.py`: quantile prediction intervals.
- `src/house_price_prediction/explainability.py`: permutation importance, partial dependence, learning curves, and group errors.

## Syllabus and Course Outcome mapping

Course Outcome labels below are concise project alignments; use the institution's official CO wording if it differs.

| Project feature / syllabus concept | Unit | Course Outcome |
|---|---:|---|
| Simple and Multiple Linear Regression | 1 | CO1 — Select and implement supervised learning algorithms |
| Ridge and Lasso regularization | 2, 5 | CO1, CO2 — Apply regularization and evaluate model generalization |
| Decision Tree and Random Forest Regression | 2 | CO1, CO2 — Compare tree-based models and control complexity |
| KNN Regressor and KNN Classifier | 3, 5 | CO1, CO2 — Apply distance-based learning and tune it |
| Logistic Regression for price tiers | 3 | CO1 — Apply classification methods |
| Confusion Matrix, Accuracy, Precision, Recall, F1, ROC-AUC | 3, 5 | CO4 — Evaluate classification outcomes with suitable metrics |
| SVR (RBF) | 2 | CO1, CO2 — Apply margin-based regression and compare models |
| HistGradientBoosting and 10th/90th percentile intervals | 5 | CO2, CO4 — Tune estimators and communicate prediction uncertainty |
| 5-fold cross-validation, holdout testing, Randomized/Grid Search | 5 | CO2 — Validate models and select hyperparameters |
| Learning and validation curves; bias-variance and over/underfitting | 5 | CO2 — Diagnose generalization and model complexity |
| K-Means with elbow and silhouette selection | 4 | CO3 — Discover market segments without labels |
| Ward hierarchical clustering and dendrogram | 4 | CO3 — Compare hierarchical grouping structure |
| PCA visualization and explained variance | 4 | CO3 — Reduce dimensions and interpret cluster structure |
| Gaussian Mixture Model (AIC/BIC comparison) | 4 | CO3 — Compare probabilistic clustering |
| Permutation importance, partial dependence, local sensitivity | 6 | CO5 — Explain model behavior |
| Region/type error checks and limitations discussion | 6 | CO5 — Assess fairness risks and responsible use |

## Data and limitations

Training expects the columns `bhk,type,locality,region,area,status,age,price_cr`. Existing filters retain carpet areas of 150–8,000 sq ft, 1–6 BHK, and ₹2,500–₹120,000 per sq ft. Localities with fewer than three listings are pooled as `Other`; region/locality encoders are cross-fitted inside each model pipeline.

Prices are listing asking prices, not confirmed sale prices. Data coverage and error can vary across places and property types. Rare localities have less reliable estimates. Model outputs are informational and must not be used as the sole basis for financial decisions.
