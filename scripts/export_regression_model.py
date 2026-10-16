#!/usr/bin/env python
"""
Model Export Script — Seoul Bike Sharing Demand Prediction (Regression Track)
=============================================================================
Replicates the preprocessing and feature engineering from the regression track
(notebooks/py_files/regression.py) and trains a Gradient Boosting Regressor
(Algorithm #8 in regression track), saving the complete sklearn Pipeline and
evaluation metrics as reusable artifacts.

Usage:
    python scripts/export_regression_model.py

Output:
    models/best_regression_pipeline.joblib  -- serialized Pipeline
    models/regression_metadata.json         -- metrics, feature schema, version info
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set LOKY_MAX_CPU_COUNT for Windows stability
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "SeoulBikeData.csv"
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_regression_pipeline.joblib"
META_PATH = MODEL_DIR / "regression_metadata.json"


def feature_engineer_bike(df: pd.DataFrame) -> pd.DataFrame:
    """Apply exact feature engineering from regression.py Phase 3."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["rush_hour"] = df.apply(
        lambda r: 1 if (r["is_weekend"] == 0 and r["hour"] in [7, 8, 9, 17, 18, 19]) else 0,
        axis=1,
    )
    df["temp_humidity_idx"] = df["temperature"] * (df["humidity"] / 100.0)
    return df


def main():
    print("=" * 70)
    print("MODEL EXPORT: Seoul Bike Sharing Demand -- Gradient Boosting Regressor")
    print("=" * 70)

    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        sys.exit(1)

    raw_bike_df = pd.read_csv(DATA_PATH, encoding="unicode_escape")
    raw_bike_df.columns = [
        "date", "rented_bike_count", "hour", "temperature", "humidity",
        "wind_speed", "visibility", "dew_point_temperature", "solar_radiation",
        "rainfall", "snowfall", "seasons", "holiday", "functioning_day",
    ]
    print(f"Raw dataset loaded: {raw_bike_df.shape[0]} rows, {raw_bike_df.shape[1]} columns")

    # Filter out non-functioning days per regression.py Step 2
    bike_clean = raw_bike_df[raw_bike_df["functioning_day"] == "Yes"].copy()
    print(f"Filtered functioning days: {len(bike_clean)} rows")

    # Feature engineering
    bike_clean = feature_engineer_bike(bike_clean)

    # Features and target definition per regression.py Step 4
    X = bike_clean.drop(columns=["date", "datetime", "rented_bike_count", "functioning_day"])
    y = bike_clean["rented_bike_count"]

    cat_cols = ["seasons", "holiday"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    print(f"Train split: {X_train.shape[0]} rows | Test split: {X_test.shape[0]} rows")

    # Preprocessing ColumnTransformer per regression.py Step 6
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
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\nTraining Gradient Boosting Regressor...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # Evaluation
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{'Metric':<25} {'Score':>10}")
    print("-" * 37)
    print(f"{'R-squared (R2)':<25} {r2:>10.4f}")
    print(f"{'RMSE (hourly rentals)':<25} {rmse:>10.2f}")
    print(f"{'MAE (hourly rentals)':<25} {mae:>10.2f}")

    # Save artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipeline, MODEL_PATH, compress=3)
    print(f"\nPipeline saved --> {MODEL_PATH}")
    print(f"  File size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")

    import sklearn

    metadata = {
        "model_name": "Gradient Boosting Regressor",
        "track": "Regression",
        "dataset_name": "Seoul Bike Sharing Demand (SeoulBikeData.csv)",
        "target_variable": "rented_bike_count (hourly count of rented bikes)",
        "model_params": {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "outlier_filter": "functioning_day == 'Yes'",
            "feature_engineering": [
                "datetime = pd.to_datetime(date, format='%d/%m/%Y')",
                "day_of_week = datetime.dt.dayofweek",
                "is_weekend = (day_of_week >= 5)",
                "rush_hour = (is_weekend == 0 and hour in [7, 8, 9, 17, 18, 19])",
                "temp_humidity_idx = temperature * (humidity / 100.0)",
            ],
            "numerical_scaler": "StandardScaler",
            "categorical_encoder": "OneHotEncoder(drop='first')",
        },
        "feature_schema": {
            "numerical": num_cols,
            "categorical": cat_cols,
            "all_input_columns": list(X.columns),
        },
        "evaluation_metrics": {
            "r2_score": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mae": round(float(mae), 2),
        },
        "dataset": {
            "total_rows": len(raw_bike_df),
            "functioning_rows": len(bike_clean),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
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
    print("EXPORT COMPLETE -- Regression model ready for Streamlit deployment")
    print("=" * 70)


if __name__ == "__main__":
    main()
