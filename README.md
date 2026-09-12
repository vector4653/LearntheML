# B.Tech Machine Learning Capstone Project - Review 1

## Overview
This repository contains the end-to-end Machine Learning pipelines developed for **Review 1** of the B.Tech Machine Learning Capstone Project. The project spans two foundational machine learning tracks:

1. **Regression Track**: Seoul Bike Sharing Demand Prediction (`notebooks/regression.ipynb`)
2. **Classification Track (Part A)**: Bank Marketing Term Deposit Subscription Prediction (`notebooks/classification.ipynb`)

---

## Repository Structure
```text
LearntheML/
├── data/                  # Raw and processed datasets
│   ├── SeoulBikeData.csv  # Seoul Bike Sharing Demand dataset
│   └── bank/              # Bank Marketing dataset (bank-full.csv, bank.csv)
├── notebooks/             # Executable Jupyter Notebooks
│   ├── regression.ipynb   # Complete Regression pipeline (10 algorithms, tuning, eval)
│   └── classification.ipynb # Complete Classification Part A pipeline (5 algorithms, eval)
├── models/                # Serialized trained model artifacts
├── build_plan.md          # Multi-phase engineering roadmap
├── progress.md            # Real-time task tracking & milestone checklist
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

---

## Technical Stack & Constraints
- **Language**: Python 3.10+
- **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Reproducibility**: Global `random_state=42` enforced across all dataset splits, encoders, and estimators.
- **Data Leakage Prevention**: Strict separation of training and testing data. Encoders and scalers are fitted exclusively on the training splits and subsequently transformed on evaluation sets.
- **Evaluation Standards**:
  - Regression: $R^2$, RMSE, MAE, 5-Fold Cross-Validation, Residual & Feature Importance diagnostics.
  - Classification: Accuracy, Weighted F1-Score, Confusion Matrices, Comparative Baselines.

---

## Setup Instructions

### 1. Clone & Environment Setup
```bash
git clone <repo-url>
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
