import sys
from pathlib import Path

import pandas as pd
import pytest


# Add the src directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

import predict


def create_valid_sample():
    return pd.DataFrame(
        [
            {
                "Gender": 1,
                "age": 50,
                "currentSmoker": 0,
                "cigsPerDay": 0,
                "BPMeds": 0,
                "diabetes": 0,
                "totChol": 220,
                "sysBP": 130,
                "diaBP": 85,
                "BMI": 25.0,
                "heartRate": 75,
                "glucose": 85,
            }
        ]
    )


def test_model_can_be_loaded():
    model = predict.load_model()

    assert model is not None


def test_valid_sample_prediction():
    sample = create_valid_sample()

    result = predict.predict(sample)

    assert result.shape == (1,)
    assert result.dtype.kind in {"i", "u"}
    assert result[0] in {0, 1}


def test_missing_required_feature_is_rejected():
    sample = create_valid_sample()

    sample = sample.drop(
        columns=["glucose"]
    )

    with pytest.raises(
        ValueError,
        match="Input is missing required features: glucose"
    ):
        predict.predict(sample)