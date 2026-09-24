from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("model_package/model.joblib")

EXPECTED_FEATURES = [
    "Gender",
    "age",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def validate_input(data):
    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Input is missing required features: "
            + ", ".join(missing_features)
        )


def predict(data):
    validate_input(data)

    model = load_model()

    features = data[EXPECTED_FEATURES]

    predictions = model.predict(features)

    return predictions


if __name__ == "__main__":
    sample = pd.DataFrame(
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

    result = predict(sample)

    print("Prediction:", result)