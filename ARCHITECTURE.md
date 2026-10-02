# Architecture Specification (`architecture.md`)

## 1. System Overview
The system follows a decoupled, three-tier machine learning lifecycle:
1. **Offline Training & Validation Pipeline:** Ingests raw Mumbai real estate listings, conducts feature transformation, trains syllabus-compliant regression models, and serializes the best performing estimator.
2. **Model Registry & Persistence Tier:** Stores the serialized Scikit-Learn `Pipeline`, categorical encoders, and feature column signatures.
3. **Serving & Inference UI Tier:** An interactive Streamlit web dashboard providing real-time property valuation and diagnostic model visualizations.

---

## 2. Technology Stack

| Layer | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language environment |
| **Data Manipulation** | `pandas`, `numpy` | Tabular data wrangling, matrix manipulation, vectorization |
| **Machine Learning** | `scikit-learn` | Regressors, transformations, clustering, metrics, hyperparameter tuning |
| **Model Explainability** | `shap` | Model interpretability (XAI) feature attribution |
| **Data Visualization** | `matplotlib`, `seaborn` | Actual vs. Predicted plots, residual histograms, correlation matrices |
| **Artifact Persistence**| `joblib` | Serialization of trained models and `ColumnTransformer` pipelines |
| **Serving & UI** | `streamlit` | Lightweight web dashboard for model interaction and analytics |

---

## 3. Dataset Specifications

* **Primary Dataset:** Mumbai House Price Dataset (~70,000 Property Records, Kaggle).
* **Coverage:** Residential properties across Mumbai City, Western Suburbs, Central Suburbs, Harbour, Navi Mumbai, and Thane.
* **Key Feature Schema:**
  * `area`: Built-up / Carpet area in square feet (Continuous numeric).
  * `bedroom_num`: Number of bedrooms / BHK (Discrete integer).
  * `bathroom_num`: Number of bathrooms (Discrete integer).
  * `balcony_num`: Number of balconies (Discrete integer).
  * `locality`: Micro-market neighborhood, e.g., Andheri West, Thane West, Dadar (Categorical).
  * `property_type`: Apartment, Independent House, Studio Apartment (Categorical).
  * `furnished`: Unfurnished, Semi-Furnished, Fully Furnished (Categorical / Ordinal).
  * `age`: Building age category (`New`, `Resale`, or `Unknown`), requested for ready-to-move properties.
  * `total_floors`: Building height in stories (Discrete integer).
  * `latitude`, `longitude`: Spatial coordinates (Continuous float).
  * `price_cr` *(Target)*: Final price in Crores.

---

## 4. Syllabus-Aligned Algorithms & Mathematical Formulation

The project incorporates algorithms and theoretical modules directly from the syllabus:

### 4.1 Supervised Learning Regressors (Module 3)
1. **Simple Linear Regression (Baseline):**
   Models price as a single-variable function of property area:
   $$\hat{y} = \beta_0 + \beta_1 x_{\text{area}}$$
2. **Multiple Linear Regression (Multivariate Baseline):**
   Accounts for simultaneous linear effects across all continuous and encoded predictors:
   $$\hat{y} = \beta_0 + \sum_{i=1}^{p} \beta_i x_i$$
3. **Regularized Linear Models (Ridge & Lasso Regression):**
   Applied to mitigate multicollinearity between area, rooms, and floor level:
   * **Ridge (L2):** Minimizes $\text{RSS} + \lambda \sum_{j=1}^{p} \beta_j^2$
   * **Lasso (L1):** Minimizes $\text{RSS} + \lambda \sum_{j=1}^{p} \vert{}\beta_j\vert{}$ (enables sparse feature selection)
4. **Random Forest Regression (Non-Linear Ensemble):**
   Constructs a bootstrap ensemble of $B$ decorrelated decision trees, aggregating predictions via averaging to lower variance:
   $$\hat{f}_{\text{rf}}^B(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$

### 4.2 Unsupervised Learning & Feature Generation (Module 4)
* **K-Means Clustering:** Applied to `latitude` and `longitude` coordinates to automatically group micro-markets into $K$ spatial clusters, creating a synthetic `cluster_id` feature:
  $$J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \vert{}\vert{}x_i - \mu_j\vert{}\vert{}^2$$
* **Principal Component Analysis (PCA) (Optional Comparative Study):** Used for dimensionality reduction across one-hot encoded amenity flags.

### 4.3 Model Optimization & Validation (Module 5)
* **K-Fold Cross-Validation:** Stratified/standard $5$-fold partitioning across training splits to assess bias-variance trade-offs.
* **Hyperparameter Tuning:** `GridSearchCV` applied to Random Forest parameters (`n_estimators`: [100, 200], `max_depth`: [10, 20, None], `min_samples_split`: [2, 5]).

### 4.4 Model Interpretability & Ethics (Module 6)
* **Explainable AI (XAI):** Integration of SHAP (Shapley Additive exPlanations) or Tree Feature Importance to display feature attributions, confirming predictions are driven by realistic physical properties rather than spurious correlations.

---

## 5. Evaluation & Diagnostic Framework

Each trained model is evaluated on a held-out test split ($20\%$) using:
* **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \vert{}y_i - \hat{y}_i\vert{}$$
* **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$
* **Coefficient of Determination ($R^2$ Score):**
  $$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

### Diagnostic Visualizations Generated
1. **Actual vs. Predicted Scatter Plot:** Evaluates alignment against the ideal $45^\circ$ diagonal parity line.
2. **Residual Distribution Histogram:** Verifies whether prediction residuals $\epsilon_i = y_i - \hat{y}_i$ are normally distributed around zero.
3. **Price Distribution Plot:** Overlays log-transformed target distributions before and after outlier removal.

---

## 6. End-to-End Pipeline Architecture Diagram