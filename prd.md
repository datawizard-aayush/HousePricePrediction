# Product Requirements Document (PRD)

## Project Title
Machine Learning-Based House Price Prediction System (Mumbai Real Estate)

---

## 1. Document Overview
* **Status:** Draft / Ready for Review
* **Target Audience:** College Project Evaluators, Academic Faculty, Prospective Homebuyers, Real Estate Analysts

---

## 2. Goals & Objectives
* **Primary Objective:** Deliver an accurate, data-backed regression model predicting residential flat/apartment prices across Mumbai localities.
* **Secondary Objective:** Benchmark linear versus non-linear tree-based regression models using standard statistical metrics ($R^2$, MAE, RMSE).
* **Tertiary Objective:** Expose the pipeline via a clean, interactive user interface allowing users to input property specs and receive an immediate price estimate with model explanations.

---

## 3. User Personas
* **Persona A (Homebuyer / Investor):** Wants to input basic property criteria (carpet area, BHK, locality, floor) to verify whether a seller's quote is fair or inflated.
* **Persona B (Academic Evaluator / Examiner):** Wants to inspect preprocessing rigor, test-train splits, benchmark metrics, diagnostic graphs, and verify that syllabus-aligned algorithms are utilized properly.

---

## 4. Key Functional Requirements

### 4.1 Data Pipeline & Feature Engineering
* **FR-1 (Ingestion):** Clean and ingest Mumbai residential property records containing spatial coordinates, structural metrics, and price targets.
* **FR-2 (Outlier Handling):** Treat extreme right-skewed prices via log-transformation ($\log(1 + y)$) and interquartile range (IQR) boundary filtering.
* **FR-3 (Encoding & Scaling):** Apply One-Hot or Target Encoding for categorical localities, and apply robust scaling to continuous variables.
* **FR-4 (Unsupervised Feature Generation):** Apply K-Means clustering on geographic coordinates (Latitude, Longitude) to engineer a spatial cluster feature.

### 4.2 Model Training & Optimization
* **FR-5 (Baseline Modeling):** Train Simple and Multiple Linear Regression models.
* **FR-6 (Regularization):** Implement Ridge (L2) and Lasso (L1) regression to minimize multicollinearity and over-parameterization.
* **FR-7 (Non-Linear Ensemble):** Train a Random Forest Regressor to capture complex feature interactions.
* **FR-8 (Hyperparameter Tuning):** Utilize Grid Search / Random Search Cross-Validation to optimize ensemble hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`).

### 4.3 Evaluation & Visualization
* **FR-9 (Metric Tracking):** Output MAE, RMSE, and $R^2$ scores across training and held-out validation sets.
* **FR-10 (Visual Diagnostics):** Render Actual vs. Predicted scatter plots, Residual Error Distribution plots, and Feature Importance bar charts.

### 4.4 User Interface & Inference
* **FR-11 (Interactive Prediction):** Provide input controls for carpet area, locality, BHK count, property type, construction status, and building age category for ready-to-move properties.
* **FR-12 (Instant Output):** Compute and display the predicted price in Crores (INR) within 1.5 seconds of submission, with holdout MAE for context.

---

## 5. Non-Functional Requirements
* **Inference Latency:** Sub-second response time for single-record prediction on CPU.
* **Code Modularity:** Separation of training scripts, preprocessing pipelines, serializable model artifacts, and UI code.
* **Platform Compatibility:** Runs on standard open-source Python stacks without proprietary dependencies.

---

## 6. Success Metrics & Acceptance Criteria
* Model achieves an $R^2$ score $\ge 0.82$ on the unseen test partition.
* Mean Absolute Error (MAE) kept within an acceptable commercial margin for property estimation.
* Zero runtime failures during real-time inference when handling typical boundary edge cases.