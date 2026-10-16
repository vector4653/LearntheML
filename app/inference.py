"""
Inference Module — Bank Marketing, Seoul Bike Sharing & Spotify Audio Clustering
================================================================================
Encapsulates model loading, input preparation, and prediction logic
for all three Capstone tracks:
  1. Classification: Bank Marketing Term Deposit Subscription
  2. Regression: Seoul Bike Sharing Hourly Demand Prediction
  3. Clustering: Spotify Audio Profile Musical Persona Discovery

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

# Classification Track paths
MODEL_PATH = _REPO_ROOT / "models" / "best_classification_pipeline.joblib"
META_PATH = _REPO_ROOT / "models" / "model_metadata.json"
DATA_PATH = _REPO_ROOT / "data" / "bank-full.csv"

# Regression Track paths
REGRESSION_MODEL_PATH = _REPO_ROOT / "models" / "best_regression_pipeline.joblib"
REGRESSION_META_PATH = _REPO_ROOT / "models" / "regression_metadata.json"
REGRESSION_DATA_PATH = _REPO_ROOT / "data" / "SeoulBikeData.csv"

# Clustering Track paths
CLUSTERING_MODEL_PATH = _REPO_ROOT / "models" / "best_clustering_pipeline.joblib"
CLUSTERING_META_PATH = _REPO_ROOT / "models" / "clustering_metadata.json"
SPOTIFY_DATA_PATH = _REPO_ROOT / "data" / "spotify-tracks-dataset.csv"

# ---------------------------------------------------------------------------
# Public Exports
# ---------------------------------------------------------------------------
__all__ = [
    "CATEGORICAL_OPTIONS",
    "RAW_FEATURE_COLS",
    "load_model",
    "load_metadata",
    "prepare_input",
    "predict",
    "BIKE_CATEGORICAL_OPTIONS",
    "BIKE_RAW_FEATURE_COLS",
    "BIKE_MODEL_ORDER",
    "load_regression_model",
    "load_regression_metadata",
    "prepare_regression_input",
    "predict_regression",
    "SPOTIFY_AUDIO_FEATURES",
    "load_clustering_model",
    "load_clustering_metadata",
    "prepare_clustering_input",
    "predict_cluster",
]

# ---------------------------------------------------------------------------
# Feature Schemas & Categorical Options
# ---------------------------------------------------------------------------
SPOTIFY_AUDIO_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# ---------------------------------------------------------------------------
# 1. Classification Track: Schema & Options (Bank Marketing)
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

RAW_FEATURE_COLS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome",
]


def _feature_engineer_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering matching classification.py Phase 3."""
    df = df.copy()
    df["pdays_contacted"] = (df["pdays"] != -1).astype(int)
    df["total_contacts"] = df["campaign"] + df["previous"]
    df["is_retired_or_student"] = df["job"].isin(["retired", "student"]).astype(int)
    df["duration_min"] = df["duration"] / 60.0
    return df


def _binary_encode_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Map binary yes/no columns to 0/1 integers."""
    df = df.copy()
    binary_map = {"no": 0, "yes": 1}
    for col in ["default", "housing", "loan"]:
        df[col] = df[col].map(binary_map)
    return df


def load_model():
    """Load the saved classification pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classification model not found at {MODEL_PATH}. "
            "Run `python scripts/export_model.py` to generate it."
        )
    return joblib.load(MODEL_PATH)


def load_metadata() -> dict[str, Any]:
    """Load the classification model metadata JSON."""
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        return json.load(f)


def prepare_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """Convert raw user inputs into the DataFrame expected by the classification pipeline."""
    row = {col: [raw_input[col]] for col in RAW_FEATURE_COLS}
    df = pd.DataFrame(row)
    df = _feature_engineer_bank(df)
    df = _binary_encode_bank(df)
    return df


def predict(pipeline, input_df: pd.DataFrame) -> dict[str, Any]:
    """Run classification inference and return predicted class and probabilities."""
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
    model_name = metadata.get("model_name", "Gradient Boosting Classifier")

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# 2. Regression Track: Schema & Options (Seoul Bike Sharing Demand)
# ---------------------------------------------------------------------------
BIKE_CATEGORICAL_OPTIONS: dict[str, list[str]] = {
    "seasons": ["Spring", "Summer", "Autumn", "Winter"],
    "holiday": ["No Holiday", "Holiday"],
    "functioning_day": ["Yes", "No"],
}

BIKE_RAW_FEATURE_COLS = [
    "hour", "temperature", "humidity", "wind_speed", "visibility",
    "dew_point_temperature", "solar_radiation", "rainfall", "snowfall",
    "seasons", "holiday",
]

BIKE_MODEL_ORDER = [
    "hour", "temperature", "humidity", "wind_speed", "visibility",
    "dew_point_temperature", "solar_radiation", "rainfall", "snowfall",
    "seasons", "holiday", "day_of_week", "is_weekend", "rush_hour",
    "temp_humidity_idx",
]


def load_regression_model():
    """Load the saved regression pipeline from disk."""
    if not REGRESSION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Regression model not found at {REGRESSION_MODEL_PATH}. "
            "Run `python scripts/export_regression_model.py` to generate it."
        )
    return joblib.load(REGRESSION_MODEL_PATH)


def load_regression_metadata() -> dict[str, Any]:
    """Load the regression model metadata JSON."""
    if not REGRESSION_META_PATH.exists():
        return {}
    with open(REGRESSION_META_PATH) as f:
        return json.load(f)


def prepare_regression_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """Convert raw user inputs into the DataFrame expected by the regression pipeline."""
    date_val = pd.to_datetime(raw_input["date"], dayfirst=True)
    day_of_week = int(date_val.dayofweek)
    is_weekend = 1 if day_of_week >= 5 else 0
    hour = int(raw_input["hour"])
    rush_hour = 1 if (is_weekend == 0 and hour in [7, 8, 9, 17, 18, 19]) else 0
    temp_humidity_idx = float(raw_input["temperature"]) * (float(raw_input["humidity"]) / 100.0)

    row = {
        "hour": [hour],
        "temperature": [float(raw_input["temperature"])],
        "humidity": [float(raw_input["humidity"])],
        "wind_speed": [float(raw_input["wind_speed"])],
        "visibility": [float(raw_input["visibility"])],
        "dew_point_temperature": [float(raw_input["dew_point_temperature"])],
        "solar_radiation": [float(raw_input["solar_radiation"])],
        "rainfall": [float(raw_input["rainfall"])],
        "snowfall": [float(raw_input["snowfall"])],
        "seasons": [str(raw_input["seasons"])],
        "holiday": [str(raw_input["holiday"])],
        "day_of_week": [day_of_week],
        "is_weekend": [is_weekend],
        "rush_hour": [rush_hour],
        "temp_humidity_idx": [temp_humidity_idx],
    }

    df = pd.DataFrame(row)
    return df[BIKE_MODEL_ORDER]


def predict_regression(pipeline, input_df: pd.DataFrame, is_functioning: bool = True) -> dict[str, Any]:
    """Run regression inference with domain filtering for maintenance days."""
    if not is_functioning:
        pred_count = 0.0
    else:
        raw_pred = pipeline.predict(input_df)[0]
        pred_count = max(0.0, float(raw_pred))

    metadata = load_regression_metadata()
    model_name = metadata.get("model_name", "Gradient Boosting Regressor")

    return {
        "predicted_count": round(pred_count, 1),
        "predicted_count_int": int(round(pred_count)),
        "is_functioning": is_functioning,
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# 3. Clustering Track: Schema & Options (Spotify Audio Features)
# ---------------------------------------------------------------------------
SPOTIFY_AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_clustering_model():
    """Load the saved clustering bundle artifact from disk."""
    if not CLUSTERING_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Clustering model bundle not found at {CLUSTERING_MODEL_PATH}. "
            "Run `python scripts/export_clustering_model.py` to generate it."
        )
    return joblib.load(CLUSTERING_MODEL_PATH)


def load_clustering_metadata() -> dict[str, Any]:
    """Load the clustering metadata JSON."""
    if not CLUSTERING_META_PATH.exists():
        return {}
    with open(CLUSTERING_META_PATH) as f:
        return json.load(f)


def prepare_clustering_input(raw_input: dict[str, Any]) -> pd.DataFrame:
    """Format raw audio feature values into a single-row DataFrame."""
    row = {feat: [float(raw_input[feat])] for feat in SPOTIFY_AUDIO_FEATURES}
    return pd.DataFrame(row)[SPOTIFY_AUDIO_FEATURES]


def predict_cluster(bundle: dict[str, Any], input_df: pd.DataFrame) -> dict[str, Any]:
    """
    Predict K-Means cluster assignment, distance to centroids,
    and 2D PCA projection coordinates.
    """
    scaler = bundle["scaler"]
    kmeans = bundle["kmeans"]
    pca = bundle["pca"]

    X_scaled = scaler.transform(input_df)
    cluster_id = int(kmeans.predict(X_scaled)[0])
    distances = kmeans.transform(X_scaled)[0]
    pca_coords = pca.transform(X_scaled)[0]

    persona_name = bundle["persona_names"].get(cluster_id, f"Cluster {cluster_id}")
    description = bundle["persona_descriptions"].get(cluster_id, "")
    exemplars = bundle["exemplars"].get(str(cluster_id), [])

    return {
        "cluster_id": cluster_id,
        "persona_name": persona_name,
        "description": description,
        "pca_x": round(float(pca_coords[0]), 3),
        "pca_y": round(float(pca_coords[1]), 3),
        "distance_to_center": round(float(distances[cluster_id]), 3),
        "exemplars": exemplars,
    }
