import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Paths and configurations
DATA_PATH = Path("data/Hypertension-risk-model-main.csv")
MODEL_DIR = Path("model_package")
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COLUMN = "Risk"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
IMPROVEMENT_MARGIN = 0.05


# Expected input features
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


# Load the dataset
def load_dataset():
    print(f"Loading dataset from: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    return df


# Validate the dataset
def validate_dataset(df):
    required_columns = EXPECTED_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset validation failed. Missing columns: " + ", ".join(missing_columns)
        )

    if df.empty:
        raise ValueError(
            "Dataset validation failed: dataset is empty."
        )

    print("Dataset validation passed.")


# Prepare features and target
def prepare_data(df):
    X = df[EXPECTED_FEATURES]
    y = df[TARGET_COLUMN]

    return X, y


# Create reproducible training/validation split
def split_data(X, y):
    return train_test_split(
        X,
        y,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# Train the baseline model
def train_baseline(X_train, y_train):
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(X_train, y_train)

    return baseline


# Build the candidate model
def build_model():
    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


# Train and evaluate the candidate model
def evaluate_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val
):
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    score = f1_score(
        y_val,
        predictions
    )

    return score


# Apply the quality gate
def quality_gate(baseline_score, model_score):
    required_score = (
        baseline_score + IMPROVEMENT_MARGIN
    )

    print(f"Baseline F1: {baseline_score:.4f}")
    print(f"Model F1:    {model_score:.4f}")
    print(f"Margin:      {IMPROVEMENT_MARGIN:.4f}")
    print(f"Required F1: {required_score:.4f}")

    if model_score < required_score:
        print("QUALITY GATE FAILED.")
        return False

    print("QUALITY GATE PASSED.")
    return True


# Save the trained model and metrics
def save_outputs(
    model,
    baseline_score,
    model_score,
    gate_passed
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    metrics = {
        "metric": "f1_score",
        "baseline_score": baseline_score,
        "model_score": model_score,
        "improvement_margin": IMPROVEMENT_MARGIN,
        "required_score": (
            baseline_score + IMPROVEMENT_MARGIN
        ),
        "gate_passed": gate_passed,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2
        )

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print(
        f"Metrics saved to: {METRICS_PATH}"
    )


# Main training pipeline
def main():
    # 1. Load dataset
    df = load_dataset()

    # 2. Validate dataset
    validate_dataset(df)

    # 3. Prepare features and target
    X, y = prepare_data(df)

    # 4. Split data
    X_train, X_val, y_train, y_val = split_data(
        X,
        y
    )

    # 5. Train baseline
    baseline = train_baseline(
        X_train,
        y_train
    )

    # 6. Evaluate baseline
    baseline_predictions = baseline.predict(
        X_val
    )

    baseline_score = f1_score(
        y_val,
        baseline_predictions
    )

    # 7. Build candidate model
    model = build_model()

    # 8. Train and evaluate candidate
    model_score = evaluate_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val
    )

    # 9. Apply quality gate
    gate_passed = quality_gate(
        baseline_score,
        model_score
    )

    # 10. Stop pipeline if quality gate fails
    if not gate_passed:
        sys.exit(1)

    # 11. Save validated model and metrics
    save_outputs(
        model,
        baseline_score,
        model_score,
        gate_passed
    )


if __name__ == "__main__":
    main()