# B.Tech Machine Learning Capstone Project

## Overview
This repository contains the end-to-end Machine Learning pipelines developed for the B.Tech Machine Learning Capstone Project. The project spans three foundational machine learning tracks:

1. **Regression Track**: Seoul Bike Sharing Demand Prediction (`notebooks/regression.ipynb`)
2. **Classification Track**: Bank Marketing Term Deposit Subscription Prediction (`notebooks/classification.ipynb`)
3. **Clustering Track**: K-Means and Agglomerative Hierarchical Clustering

---

## ⚡ Interactive Dual-Track Prediction Application (Review 2 Bonus)

A unified **Streamlit web application** that provides interactive graphical inference for both capstone prediction tracks:

1. **🏦 Track 1 — Classification:** Predicts whether a bank client will subscribe to a term deposit (`yes` / `no`) with class probabilities.
2. **🚲 Track 2 — Regression:** Predicts the hourly count of rented bikes in Seoul based on calendar schedule, commuter dynamics, and atmospheric conditions.

### Highlights
- **Unified Interface:** Sidebar track selector allowing evaluators to switch seamlessly between Classification and Regression.
- **Production Preprocessing Pipelines:** Complete scikit-learn `Pipeline` and `ColumnTransformer` artifacts combining numerical scaling, categorical encoding, and custom domain feature engineering.
- **Strict Leakage Prevention:** All scalers and encoders fitted exclusively on training splits (80/20 stratified/random split, `random_state=42`).
- **Domain Logic:** Deterministic structural filter handling for Seoul Bike station maintenance days (`functioning_day == 'No'` yields 0 rentals).
- **Full Model Transparency:** Embedded model evaluation cards, hyperparameter disclosures, and interactive sample scenarios.

---

### Verified Model Performance

| Track | Dataset | Selected Model | Primary Evaluation Metrics |
|---|---|---|---|
| **Classification** | Bank Marketing (`bank-full.csv`, 45,211 rows) | **Gradient Boosting Classifier** | **Accuracy:** 91.08%<br>**ROC-AUC:** 0.9323<br>**Weighted F1:** 90.44% |
| **Regression** | Seoul Bike Sharing (`SeoulBikeData.csv`, 8,760 rows) | **Gradient Boosting Regressor** | **$R^2$ Score:** 0.9275<br>**RMSE:** 168.70 bikes<br>**MAE:** 92.88 bikes |

---

### Quick Start (Local Execution)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export serialized model artifacts (if models/ directory is empty)
python scripts/export_model.py             # Generates classification artifact
python scripts/export_regression_model.py  # Generates regression artifact

# 3. Launch the dual-track Streamlit application
streamlit run app/streamlit_app.py
```
The app will launch at `http://localhost:8501`.

---

### Deployed Application
> **Deployment Guide:** Complete instructions for Streamlit Community Cloud are documented in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Repository Structure
```text
LearntheML/
├── app/                                    # Streamlit GUI application
│   ├── __init__.py
│   ├── streamlit_app.py                    # Main dual-track GUI entry point
│   └── inference.py                        # Classification & regression inference engine
├── data/                                   # Raw datasets
│   ├── SeoulBikeData.csv                   # Seoul Bike Sharing Demand dataset
│   ├── bank-full.csv                       # Bank Marketing dataset (45,211 records)
│   ├── bank.csv                            # Bank Marketing (subset)
│   └── bank-names.txt                      # Dataset schema documentation
├── models/                                 # Serialized trained model artifacts
│   ├── best_classification_pipeline.joblib # Trained Gradient Boosting Classifier (869 KB)
│   ├── model_metadata.json                 # Classification metrics & schema
│   ├── best_regression_pipeline.joblib     # Trained Gradient Boosting Regressor (551 KB)
│   └── regression_metadata.json            # Regression metrics & schema
├── notebooks/                              # Executable Jupyter Notebooks
│   ├── regression.ipynb                    # Complete Regression pipeline (10 algorithms, tuning)
│   ├── classification.ipynb                # Complete Classification pipeline (algorithms, eval)
│   └── py_files/                           # Script-first development files
├── scripts/                                # Standalone model export utilities
│   ├── export_model.py                     # Classification pipeline export
│   └── export_regression_model.py          # Regression pipeline export
├── tests/                                  # Automated test suite
│   └── test_inference.py                   # Dual-track smoke & prediction tests (11 passing)
├── .streamlit/                             # Streamlit runtime configuration
│   └── config.toml
├── build_plan.md                           # Multi-phase engineering roadmap
├── progress.md                             # Task tracking & milestones
├── requirements.txt                        # Pinned dependencies
├── DEPLOYMENT.md                           # Cloud deployment guide
└── README.md                               # Project documentation
```

---

## Technical Stack & Constraints
- **Language**: Python 3.10+
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn==1.9.1`, `joblib`, `matplotlib`, `seaborn`
- **GUI & Deployment**: `streamlit>=1.30.0`
- **Reproducibility**: Global `random_state=42` enforced across all dataset splits, encoders, and estimators.
- **Data Leakage Prevention**: Strict separation of training and testing data. Encoders and scalers are fitted exclusively on the training splits and subsequently transformed on evaluation sets.
- **Evaluation Standards**:
  - Regression: $R^2$, RMSE, MAE, Residual & Feature Importance diagnostics.
  - Classification: Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score, ROC-AUC.
