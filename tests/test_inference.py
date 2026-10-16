"""
Smoke tests for model loading and prediction (Classification & Regression tracks).
Run with: python -m pytest tests/test_inference.py -v
Or simply: python tests/test_inference.py
"""

import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.inference import (
    BIKE_CATEGORICAL_OPTIONS,
    BIKE_MODEL_ORDER,
    CATEGORICAL_OPTIONS,
    RAW_FEATURE_COLS,
    load_metadata,
    load_model,
    load_regression_metadata,
    load_regression_model,
    predict,
    predict_regression,
    prepare_input,
    prepare_regression_input,
)


def _sample_classification_input() -> dict:
    """Return a valid sample input dictionary for Bank Marketing."""
    return {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "duration": 200,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }


def _sample_regression_input() -> dict:
    """Return a valid sample input dictionary for Seoul Bike Demand."""
    return {
        "date": "15/06/2018",
        "hour": 18,
        "temperature": 25.5,
        "humidity": 50,
        "wind_speed": 2.1,
        "visibility": 1800,
        "dew_point_temperature": 14.2,
        "solar_radiation": 1.8,
        "rainfall": 0.0,
        "snowfall": 0.0,
        "seasons": "Summer",
        "holiday": "No Holiday",
        "functioning_day": "Yes",
    }


# ===========================================================================
# Classification Track Tests
# ===========================================================================
def test_classification_model_loads():
    """Test that the saved classification artifact loads without error."""
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    print("PASS: Classification model loads successfully")


def test_classification_metadata_loads():
    """Test that classification metadata JSON loads and contains expected keys."""
    meta = load_metadata()
    assert "model_name" in meta
    assert "evaluation_metrics" in meta
    assert "feature_schema" in meta
    print("PASS: Classification metadata loads successfully")


def test_classification_prepare_input():
    """Test that classification input preparation produces a valid DataFrame."""
    raw = _sample_classification_input()
    df = prepare_input(raw)
    assert len(df) == 1
    assert "pdays_contacted" in df.columns
    assert "total_contacts" in df.columns
    assert "is_retired_or_student" in df.columns
    assert "duration_min" in df.columns
    assert df["default"].iloc[0] in (0, 1)
    assert df["housing"].iloc[0] in (0, 1)
    assert df["loan"].iloc[0] in (0, 1)
    print("PASS: Classification input preparation works correctly")


def test_classification_prediction_output():
    """Test end-to-end classification prediction returns valid structure."""
    model = load_model()
    raw = _sample_classification_input()
    df = prepare_input(raw)
    result = predict(model, df)

    assert result["predicted_class"] in ("yes", "no")
    assert result["model_name"] is not None

    if result["probabilities"] is not None:
        assert "yes" in result["probabilities"]
        assert "no" in result["probabilities"]
        assert 0 <= result["probabilities"]["yes"] <= 1
        assert 0 <= result["probabilities"]["no"] <= 1
        total = result["probabilities"]["yes"] + result["probabilities"]["no"]
        assert abs(total - 1.0) < 0.01

    print(f"PASS: Classification Prediction = {result['predicted_class']}, "
          f"P(yes) = {result['probabilities']['yes'] if result['probabilities'] else 'N/A'}")


def test_classification_all_categorical_options():
    """Verify that all classification categorical options produce valid predictions."""
    model = load_model()
    raw = _sample_classification_input()

    for cat_col, options in CATEGORICAL_OPTIONS.items():
        for opt in options:
            raw_copy = raw.copy()
            raw_copy[cat_col] = opt
            df = prepare_input(raw_copy)
            result = predict(model, df)
            assert result["predicted_class"] in ("yes", "no"), (
                f"Invalid prediction for {cat_col}={opt}"
            )

    print("PASS: All classification categorical options produce valid predictions")


# ===========================================================================
# Regression Track Tests
# ===========================================================================
def test_regression_model_loads():
    """Test that the saved regression artifact loads without error."""
    model = load_regression_model()
    assert model is not None
    assert hasattr(model, "predict")
    print("PASS: Regression model loads successfully")


def test_regression_metadata_loads():
    """Test that regression metadata JSON loads and contains expected keys."""
    meta = load_regression_metadata()
    assert "model_name" in meta
    assert "evaluation_metrics" in meta
    assert "feature_schema" in meta
    assert meta["evaluation_metrics"]["r2_score"] > 0.85
    print("PASS: Regression metadata loads successfully")


def test_regression_prepare_input():
    """Test that regression input preparation produces exact columns and engineered features."""
    raw = _sample_regression_input()
    df = prepare_regression_input(raw)
    assert len(df) == 1
    assert list(df.columns) == BIKE_MODEL_ORDER
    assert df["day_of_week"].iloc[0] == 4  # 15/06/2018 was Friday (index 4)
    assert df["is_weekend"].iloc[0] == 0
    assert df["rush_hour"].iloc[0] == 1   # Friday at 18:00 is rush hour
    print("PASS: Regression input preparation works correctly")


def test_regression_prediction_output():
    """Test regression prediction returns a positive integer count."""
    model = load_regression_model()
    raw = _sample_regression_input()
    df = prepare_regression_input(raw)
    result = predict_regression(model, df, is_functioning=True)

    assert "predicted_count" in result
    assert "predicted_count_int" in result
    assert result["predicted_count"] >= 0.0
    assert result["predicted_count_int"] >= 0
    print(f"PASS: Regression Prediction = {result['predicted_count_int']} bikes (rush hour summer)")


def test_regression_non_functioning_day():
    """Test that non-functioning day deterministically returns 0 rentals."""
    model = load_regression_model()
    raw = _sample_regression_input()
    raw["functioning_day"] = "No"
    df = prepare_regression_input(raw)
    result = predict_regression(model, df, is_functioning=False)

    assert result["predicted_count"] == 0.0
    assert result["predicted_count_int"] == 0
    print("PASS: Regression non-functioning day returns 0 bikes deterministically")


def test_regression_all_seasons_and_holidays():
    """Verify that all season and holiday values produce valid positive predictions."""
    model = load_regression_model()
    raw = _sample_regression_input()

    for season in BIKE_CATEGORICAL_OPTIONS["seasons"]:
        for holiday in BIKE_CATEGORICAL_OPTIONS["holiday"]:
            raw_copy = raw.copy()
            raw_copy["seasons"] = season
            raw_copy["holiday"] = holiday
            df = prepare_regression_input(raw_copy)
            result = predict_regression(model, df, is_functioning=True)
            assert result["predicted_count"] >= 0.0

    print("PASS: All seasons and holiday combinations produce valid predictions")


if __name__ == "__main__":
    print("=" * 60)
    print("Running dual-track inference smoke tests...")
    print("=" * 60)
    test_classification_model_loads()
    test_classification_metadata_loads()
    test_classification_prepare_input()
    test_classification_prediction_output()
    test_classification_all_categorical_options()

    test_regression_model_loads()
    test_regression_metadata_loads()
    test_regression_prepare_input()
    test_regression_prediction_output()
    test_regression_non_functioning_day()
    test_regression_all_seasons_and_holidays()
    print("=" * 60)
    print("ALL DUAL-TRACK TESTS PASSED")
    print("=" * 60)
