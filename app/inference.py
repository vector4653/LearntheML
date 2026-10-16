"""
Inference Module — Bank Marketing Term Deposit Prediction
==========================================================
Encapsulates model loading, input preparation, and prediction logic.
Kept separate from the Streamlit UI for maintainability and testability.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to repo root, resolved at import time)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
MODEL_PATH = _REPO_ROOT / "models" / "best_classification_pipeline.joblib"
META_PATH = _REPO_ROOT / "models" / "model_metadata.json"
DATA_PATH = _REPO_ROOT / "data" / "bank-full.csv"

# ---------------------------------------------------------------------------
# Dataset schema — derived from bank-full.csv and classification.py
# ---------------------------------------------------------------------------
CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single"],
    "education": ["primary", "secondary", "tertiary", "unknown"],
    "contact": ["cellular", "telephone", "unknown"],
    "month": [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
    "poutcome": ["failure", "other", "success", "unknown"],
}

# Original raw columns (before feature engineering) in the order they appear
RAW_FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome",
]


# ---------------------------------------------------------------------------
# Feature engineering — mirrors scripts/export_model.py exactly
# ---------------------------------------------------------------------------
def _feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as the team's pipeline."""
    df = df.copy()
    df["pdays_contacted"] = (df["pdays"] != -1).astype(int)
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["is_retired_or_student"] = df["job"].isin(["retired", "student"]).astype(int)
    df["duration_min"] = df["duration"] / 60.0
    return df


def _binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary yes/no columns to 0/1."""
    df = df.copy()
    binary_map = {"no": 0, "yes": 1}
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map(binary_map)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_model():
    """Load the saved scikit-learn Pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run `python scripts/export_model.py` to generate it."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    """Load the model metadata JSON."""
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)


def prepare_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """
    Convert a dictionary of raw user inputs into a DataFrame ready for
    the trained pipeline.

    Parameters
    ----------
    raw_input : dict
        Keys must match RAW_FEATURE_COLS. Binary columns (default, housing,
        loan) should be 'yes'/'no' strings.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with feature-engineered columns, matching the
        schema expected by the saved Pipeline.
    """
    # Build a single-row DataFrame in the correct column order
    row = {col: [raw_input[col]] for col in RAW_FEATURE_COLS}
    df = pd.DataFrame(row)

    # Apply the same preprocessing as training
    df = _feature_engineer(df)
    df = _binary_encode(df)

    return df


def predict(pipeline, input_df: pd.DataFrame) -> dict[str, Any]:
    """
    Run inference and return structured results.

    Returns
    -------
    dict with keys:
        - predicted_class: 'yes' or 'no'
        - probabilities: {'no': float, 'yes': float} or None
        - model_name: str
    """
    pred = pipeline.predict(input_df)[0]
    predicted_class = "yes" if pred == 1 else "no"

    probabilities = None
    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(input_df)[0]
            probabilities = {
                "no": round(float(proba[0]), 4),
                "yes": round(float(proba[1]), 4),
            }
        except Exception:
            pass

    metadata = load_metadata()
    model_name = metadata.get("model_name", "Classification Model")

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_name": model_name,
    }
