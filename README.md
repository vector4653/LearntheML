# B.Tech Machine Learning Capstone Project

## Overview
This repository contains the end-to-end Machine Learning pipelines developed for the B.Tech Machine Learning Capstone Project. The project spans three foundational machine learning tracks:

1. **Regression Track**: Seoul Bike Sharing Demand Prediction (`notebooks/regression.ipynb`)
2. **Classification Track**: Bank Marketing Term Deposit Subscription Prediction (`notebooks/classification.ipynb`)
3. **Clustering Track**: K-Means and Agglomerative Hierarchical Clustering

---

## 🏦 Interactive Prediction Application (Review 2 Bonus)

A polished **Streamlit web application** that predicts whether a bank client will subscribe to a term deposit, using a trained **Gradient Boosting Classifier**.

### Features
- Enter customer profile, financial information, and campaign details via an interactive form.
- Receive a prediction (`yes` / `no`) with model-estimated probabilities.
- View verified model evaluation metrics (Accuracy: 91.08%, ROC-AUC: 93.23%).
- Full transparency section with model configuration, methodology notes, and environment versions.

### Quick Start (Local)
```bash
# Install dependencies
pip install -r requirements.txt

# Generate the model artifact (first time only)
python scripts/export_model.py

# Launch the application
streamlit run app/streamlit_app.py
```
The app opens at `http://localhost:8501`.

### Deployed Application
> **Deployment instructions:** See [DEPLOYMENT.md](DEPLOYMENT.md) for Streamlit Community Cloud deployment steps.

### Selected Model
| Metric | Score |
|---|---|
| Accuracy | 91.08% |
| Weighted Precision | 90.21% |
| Weighted Recall | 91.08% |
| Weighted F1-Score | 90.44% |
| ROC-AUC | 0.9323 |

**Model:** Gradient Boosting Classifier (n_estimators=200, max_depth=5, learning_rate=0.1)
**Preprocessing:** StandardScaler + OneHotEncoder (drop='first') via sklearn Pipeline.
**Dataset:** UCI Bank Marketing (`bank-full.csv`) — 45,211 records, 80/20 stratified split, random_state=42.

---

## Repository Structure
```text
LearntheML/
├── app/                         # Streamlit GUI application
│   ├── streamlit_app.py         # Main application entry point
│   └── inference.py             # Model loading & prediction logic
├── data/                        # Raw datasets
│   ├── SeoulBikeData.csv        # Seoul Bike Sharing Demand dataset
│   ├── bank-full.csv            # Bank Marketing dataset
│   ├── bank.csv                 # Bank Marketing (subset)
│   └── bank-names.txt           # Dataset documentation
├── models/                      # Serialized trained model artifacts
│   ├── best_classification_pipeline.joblib
│   └── model_metadata.json
├── notebooks/                   # Executable Jupyter Notebooks
│   ├── regression.ipynb         # Complete Regression pipeline
│   ├── classification.ipynb     # Complete Classification pipeline
│   └── py_files/                # Script-first development files
├── scripts/                     # Utility scripts
│   └── export_model.py          # Model training & export script
├── tests/                       # Automated tests
│   └── test_inference.py        # Inference smoke tests
├── build_plan.md                # Multi-phase engineering roadmap
├── progress.md                  # Task tracking & milestones
├── requirements.txt             # Environment dependencies
├── DEPLOYMENT.md                # Streamlit Cloud deployment guide
└── README.md                    # Project documentation
```

---

## Technical Stack & Constraints
- **Language**: Python 3.10+
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **GUI**: `streamlit`
- **Reproducibility**: Global `random_state=42` enforced across all dataset splits, encoders, and estimators.
- **Data Leakage Prevention**: Strict separation of training and testing data. Encoders and scalers are fitted exclusively on the training splits and subsequently transformed on evaluation sets.
- **Evaluation Standards**:
  - Regression: $R^2$, RMSE, MAE, 5-Fold Cross-Validation, Residual & Feature Importance diagnostics.
  - Classification: Accuracy, Weighted F1-Score, Confusion Matrices, Comparative Baselines.

---

## Setup Instructions

### 1. Clone & Environment Setup
```bash
git clone https://github.com/vector4653/LearntheML.git
cd LearntheML
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Running Notebooks
Launch JupyterLab or VS Code notebook interface:
```bash
jupyter lab
```
Navigate to `notebooks/regression.ipynb` or `notebooks/classification.ipynb` and run cells top-to-bottom.

### 3. Running the Prediction App
```bash
# Generate model (if models/ directory is empty)
python scripts/export_model.py

# Start the app
streamlit run app/streamlit_app.py
```
