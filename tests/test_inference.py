"""
Smoke tests for model loading and prediction.
Run with: python -m pytest tests/test_inference.py -v
Or simply: python tests/test_inference.py
"""

import sys
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.inference import (
    CATEGORICAL_OPTIONS,
    RAW_FEATURE_COLS,
    load_metadata,
    load_model,
    predict,
    prepare_input,
)


def _sample_input() -> dict:
    """Return a valid sample input dictionary."""
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


def test_model_loads():
    """Test that the saved model artifact loads without error."""
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    print("PASS: Model loads successfully")


def test_metadata_loads():
    """Test that metadata JSON loads and contains expected keys."""
    meta = load_metadata()
    assert "model_name" in meta
    assert "evaluation_metrics" in meta
    assert "feature_schema" in meta
    print("PASS: Metadata loads successfully")


def test_prepare_input():
    """Test that input preparation produces a valid DataFrame."""
    raw = _sample_input()
    df = prepare_input(raw)
    assert len(df) == 1
    # Should have the engineered features
    assert "pdays_contacted" in df.columns
    assert "total_contacts" in df.columns
    assert "is_retired_or_student" in df.columns
    assert "duration_min" in df.columns
    # Binary columns should be 0/1
    assert df["default"].iloc[0] in (0, 1)
    assert df["housing"].iloc[0] in (0, 1)
    assert df["loan"].iloc[0] in (0, 1)
    print("PASS: Input preparation works correctly")


def test_prediction_output():
    """Test end-to-end prediction returns valid structure."""
    model = load_model()
    raw = _sample_input()
    df = prepare_input(raw)
    result = predict(model, df)

    assert result["predicted_class"] in ("yes", "no")
    assert result["model_name"] is not None

    if result["probabilities"] is not None:
        assert "yes" in result["probabilities"]
        assert "no" in result["probabilities"]
        assert 0 <= result["probabilities"]["yes"] <= 1
        assert 0 <= result["probabilities"]["no"] <= 1
        # Probabilities should sum to ~1
        total = result["probabilities"]["yes"] + result["probabilities"]["no"]
        assert abs(total - 1.0) < 0.01

    print(f"PASS: Prediction = {result['predicted_class']}, "
          f"P(yes) = {result['probabilities']['yes'] if result['probabilities'] else 'N/A'}")


def test_all_categorical_options():
    """Verify that all categorical options produce valid predictions."""
    model = load_model()
    raw = _sample_input()

    for cat_col, options in CATEGORICAL_OPTIONS.items():
        for opt in options:
            raw_copy = raw.copy()
            raw_copy[cat_col] = opt
            df = prepare_input(raw_copy)
            result = predict(model, df)
            assert result["predicted_class"] in ("yes", "no"), (
                f"Invalid prediction for {cat_col}={opt}"
            )

    print("PASS: All categorical options produce valid predictions")


def test_feature_column_order():
    """Verify RAW_FEATURE_COLS matches the expected 16 columns."""
    assert len(RAW_FEATURE_COLS) == 16
    assert "y" not in RAW_FEATURE_COLS  # target should not be in features
    print("PASS: Feature column order verified")


if __name__ == "__main__":
    print("=" * 60)
    print("Running inference smoke tests...")
    print("=" * 60)
    test_model_loads()
    test_metadata_loads()
    test_prepare_input()
    test_prediction_output()
    test_all_categorical_options()
    test_feature_column_order()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
