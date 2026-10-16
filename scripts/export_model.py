#!/usr/bin/env python
"""
Model Export Script — Bank Marketing Term Deposit Prediction
=============================================================
Replicates the exact preprocessing pipeline from the classification track
(notebooks/py_files/classification.py) and trains a Gradient Boosting Classifier
(Review 2 algorithm #8), saving the complete sklearn Pipeline + evaluation
metrics as a reusable artifact.

Usage:
    python scripts/export_model.py

Output:
    models/best_classification_pipeline.joblib  — serialized Pipeline
    models/model_metadata.json                  — metrics, feature schema, version info
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Resolve paths relative to the repository root (parent of scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "bank-full.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_classification_pipeline.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"


# ---------------------------------------------------------------------------
# Feature Engineering — mirrors classification.py Phase 3 exactly
# ---------------------------------------------------------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the team's classification pipeline."""
    df = df.copy()
    df["pdays_contacted"] = (df["pdays"] != -1).astype(int)
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["is_retired_or_student"] = df["job"].isin(["retired", "student"]).astype(int)
    df["duration_min"] = df["duration"] / 60.0
    return df


def binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary yes/no columns to 0/1 integers — mirrors classification.py."""
    df = df.copy()
    binary_map = {"no": 0, "yes": 1}
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map(binary_map)
    return df


# ---------------------------------------------------------------------------
# Main Export Routine
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("MODEL EXPORT: Bank Marketing — Gradient Boosting Classifier")
    print("=" * 70)

    # 1. Load data ──────────────────────────────────────────────────────────
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    bank_df = pd.read_csv(DATA_PATH, sep=";")
    print(f"Dataset loaded: {bank_df.shape[0]} rows, {bank_df.shape[1]} columns")

    # 2. Feature engineering ────────────────────────────────────────────────
    bank_clean = feature_engineer(bank_df)
    bank_clean = binary_encode(bank_clean)

    # Encode target
    bank_clean["y"] = bank_clean["y"].map({"no": 0, "yes": 1})

    X = bank_clean.drop(columns=["y"])
    y = bank_clean["y"]

    # Define column groups — identical to classification.py
    cat_cols = ["job", "marital", "education", "contact", "month", "poutcome"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # 3. Stratified train/test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # 4. Build the full Pipeline ────────────────────────────────────────────
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"), cat_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # 5. Train ──────────────────────────────────────────────────────────────
    print("\nTraining Gradient Boosting Classifier …")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 6. Evaluate ───────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n{'Metric':<25} {'Score':>8}")
    print("-" * 35)
    print(f"{'Accuracy':<25} {accuracy:>8.4f}")
    print(f"{'Weighted Precision':<25} {precision:>8.4f}")
    print(f"{'Weighted Recall':<25} {recall:>8.4f}")
    print(f"{'Weighted F1-Score':<25} {f1:>8.4f}")
    print(f"{'ROC-AUC':<25} {roc_auc:>8.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["no", "yes"]))

    # 7. Save artifacts ─────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Pipeline saved --> {MODEL_PATH}")
    print(f"  File size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")

    # Save metadata
    import sklearn

    metadata = {
        "model_name": "Gradient Boosting Classifier",
        "model_params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "feature_engineering": [
                "pdays_contacted = (pdays != -1)",
                "total_contacts = campaign + previous",
                "is_retired_or_student = job in [retired, student]",
                "duration_min = duration / 60",
            ],
            "binary_encoded": ["default", "housing", "loan"],
            "numerical_scaler": "StandardScaler",
            "categorical_encoder": "OneHotEncoder(drop='first')",
        },
        "feature_schema": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "all_input_columns": list(X.columns),
        },
        "evaluation_metrics": {
            "accuracy": round(accuracy, 4),
            "weighted_precision": round(precision, 4),
            "weighted_recall": round(recall, 4),
            "weighted_f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
        },
        "dataset": {
            "name": "Bank Marketing (bank-full.csv)",
            "total_rows": len(bank_df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "target_distribution": {
                "no": int((y == 0).sum()),
                "yes": int((y == 1).sum()),
            },
        },
        "versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved --> {META_PATH}")

    print("\n" + "=" * 70)
    print("EXPORT COMPLETE -- model ready for Streamlit deployment")
    print("=" * 70)


if __name__ == "__main__":
    main()
