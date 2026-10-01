# Automated ML Pipeline for Hypertension Risk Classification

## Lab Assignment 4: Automating an ML Pipeline with GitHub Actions

This project implements an automated machine learning pipeline for hypertension risk classification using Python, Scikit-learn, Pytest, Git, GitHub, and GitHub Actions.

The pipeline automatically:

1. Validates the dataset.
2. Splits the data into training and validation sets.
3. Trains a baseline model.
4. Trains a candidate Random Forest model.
5. Evaluates both models using F1-score.
6. Applies a predefined quality gate.
7. Runs automated application tests.
8. Creates a model package.
9. Publishes the model package as a GitHub Actions artifact only when all required checks pass.

The main purpose of this project is to demonstrate a reproducible, automated ML workflow rather than a manually trained model.

---

## 1. Project Objective

The objective is to build an automated pipeline that ensures a trained model is not published unless it satisfies both:

- A model performance requirement.
- Required application-level tests.

The pipeline fails automatically when:

- Required dataset columns are missing.
- The model does not improve sufficiently over the baseline.
- Required application tests fail.

The final model package is published only when all required stages pass.

---

## 2. Problem Statement

Hypertension is a major health risk associated with demographic, lifestyle, and physiological factors. This project uses a tabular dataset to build a **binary classification model** that predicts the `Risk` target from patient-related features.

> This project is intended for ML pipeline experimentation and is **not a clinical diagnostic system**.

---

## 3. Dataset

**File:** `data/Hypertension-risk-model-main.csv`

The CSV was provided as the dataset for the lab experiment. It is below the assignment's 5 MB limit and is therefore stored directly in the repository.

- **4,240 rows**
- **13 columns** (12 input features + 1 target)

### Target Variable

`Risk`, with two classes:

| Risk Class | Number of Records |
|-----------:|------------------:|
| 0 | 2,923 |
| 1 | 1,317 |

The dataset is not perfectly balanced.

### Input Features

| Feature | Description |
|---|---|
| Gender | Gender indicator |
| age | Age |
| currentSmoker | Current smoking status |
| cigsPerDay | Number of cigarettes smoked per day |
| BPMeds | Blood pressure medication indicator |
| diabetes | Diabetes indicator |
| totChol | Total cholesterol |
| sysBP | Systolic blood pressure |
| diaBP | Diastolic blood pressure |
| BMI | Body Mass Index |
| heartRate | Heart rate |
| glucose | Glucose level |

**Feature naming note:** the original dataset column `male` was renamed to `Gender`. The training and prediction scripts use `Gender` consistently.

---

## 4. Machine Learning Task

**Binary classification:** the model predicts `Risk = 0` or `Risk = 1`.

The pipeline compares a simple baseline model with a Random Forest candidate model.

---

## 5. Project Structure

```text
hypertension-risk/
│
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml
│
├── data/
│   └── Hypertension-risk-model-main.csv
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_prediction.py
│
├── model_package/
│   ├── model.joblib
│   └── metrics.json
│
├── requirements.txt
│
└── .gitignore
```

| File | Purpose |
|---|---|
| `data/Hypertension-risk-model-main.csv` | Dataset used for training and validation |
| `src/train.py` | Full training pipeline: dataset loading and validation, train/validation split, baseline and candidate training, F1 calculation, quality gate, model and metrics saving |
| `src/predict.py` | Loads the trained model, validates that all required features are present, and predicts on new input |
| `tests/test_prediction.py` | Automated application tests |
| `model_package/` | Locally generated model and metrics (ignored by Git; produced by the pipeline) |
| `.github/workflows/ml-pipeline.yml` | GitHub Actions workflow automating the pipeline |
| `requirements.txt` | Python dependencies |

---

## 6. Technologies Used

- Python 3.13
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Pytest
- Git and GitHub
- GitHub Actions

---

## 7. Baseline Model

```python
DummyClassifier(strategy="most_frequent")
```

The baseline always predicts the most frequent class in the training data. It is not meant to be a useful predictor. It is a simple reference point so the candidate model must demonstrate meaningful improvement rather than just existing.

---

## 8. Candidate Model

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```

Random Forest was selected because it suits tabular classification data and can model nonlinear relationships between features. `class_weight="balanced"` helps account for the imbalance between the two target classes.

---

## 9. Data Preprocessing

Some input features contain missing values. Preprocessing uses:

```python
SimpleImputer(strategy="median")
```

The imputer is part of a Scikit-learn `Pipeline`:

```text
SimpleImputer  →  RandomForestClassifier
```

Preprocessing is fitted **only on the training data**; the validation data is never used to compute imputation values. This prevents data leakage from the validation set into training.

---

## 10. Train/Validation Split

The data is divided into **80% training** and **20% validation**:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

These settings remain fixed throughout the experiment:

- Validation size = 20%
- Random state = 42
- Stratification = enabled

A fixed random state makes the experiment reproducible, and stratification approximately preserves the class distribution in both sets.

---

## 11. Evaluation Metric: F1-Score

The primary metric is the **F1-score**, the harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Because the dataset is imbalanced (substantially more `Risk = 0` than `Risk = 1`), accuracy alone could be misleading, since a model could score well by mostly predicting the majority class. F1 considers both precision and recall, which gives a more useful measure when performance on the positive class matters.

The same metric is used for both the baseline and the candidate model.

---

## 12. Quality Gate

The improvement margin is **0.05 F1 points**. Since F1 is higher-is-better, the candidate must satisfy:

```text
Model F1 >= Baseline F1 + 0.05
```

The pipeline computes `Required F1 = Baseline F1 + 0.05`. If the candidate does not meet this, the training script exits with a non-zero status:

```python
if model_score < required_score:
    print("QUALITY GATE FAILED.")
    sys.exit(1)
```

This causes GitHub Actions to mark the training step as failed.

### Why a 0.05 margin?

It is a practical threshold requiring meaningful improvement over the baseline. Without a margin, a candidate could pass with an extremely small, potentially meaningless improvement. The margin was fixed **before** the failure/recovery demonstrations and was not changed during them.

If the improvement margin is too low, a model with only a very small improvement over the baseline may pass the quality gate, even though the improvement may not be practically meaningful. If the margin is too high, a genuinely useful model may fail the quality gate because the required improvement is too difficult to achieve. Therefore, the margin should be large enough to require meaningful improvement but not so large that it unnecessarily rejects useful models.

---

## 13. Observed Model Performance

From the successful training run:

```text
Baseline F1: 0.0000
Model F1:    0.8342
Margin:      0.0500
Required F1: 0.0500

0.8342 >= 0.0500  →  QUALITY GATE PASSED
```

The baseline F1 is 0 because the most-frequent classifier predicts only the majority class and never identifies the positive class. The Random Forest achieved an F1 of approximately **0.8342** on the fixed validation split.

---

## 14. Automated Application Tests

Pytest verifies that the trained model can be used correctly. Three tests are implemented:

| Test | What it verifies |
|---|---|
| `test_model_can_be_loaded` | The saved model package contains a valid serialized model that loads successfully |
| `test_valid_sample_prediction` | A valid sample with all required features produces a prediction of expected shape `(1,)`, an integer class, belonging to the expected classes |
| `test_missing_required_feature_is_rejected` | Removing the `glucose` feature causes a clear error: `Input is missing required features: glucose` |

Final local test run:

```text
test_model_can_be_loaded                  PASSED
test_valid_sample_prediction              PASSED
test_missing_required_feature_is_rejected PASSED

3 passed
```

---

## 15. GitHub Actions Workflow

The workflow (`.github/workflows/ml-pipeline.yml`) is triggered on every push to `main`:

```yaml
on:
  push:
    branches:
      - main
```

### Pipeline Stages

```text
Git Push
   ↓
Checkout Repository          (actions/checkout@v4)
   ↓
Set Up Python 3.13           (actions/setup-python@v5)
   ↓
Install Dependencies         (pip install -r requirements.txt)
   ↓
Train Model + Quality Gate   (python src/train.py)
   ↓
Run Application Tests        (pytest -v)
   ↓
Create Model Package
   ↓
Upload Artifact              (actions/upload-artifact@v4)
```

**Training and quality gate:** `train.py` performs dataset validation, the train/validation split, baseline and candidate training, F1 evaluation, and the quality gate. A non-zero exit fails the job.

**Application testing:** `pytest -v` runs after successful training. All tests must pass before the package is created. This prevents a model with good validation performance from being published if the application using it is broken.

**Dependencies** (`requirements.txt`): `pandas`, `numpy`, `scikit-learn`, `joblib`, `pytest`. This makes the pipeline independent of the developer's local environment.

### Model Package

```text
artifact/
├── model.joblib      # fitted preprocessing + Random Forest pipeline
├── metrics.json      # metric name, baseline score, model score, margin, required score, gate result
├── predict.py        # prediction logic
└── requirements.txt  # dependencies for using the model
```

### Artifact Publishing

The artifact is named with the workflow run number:

```text
hypertension-model-run-${{ github.run_number }}
```

For example, `hypertension-model-run-1`. This links each model package to the workflow run that produced it.

The upload step does **not** use `if: always()`, and no required step uses `continue-on-error: true`. The artifact is therefore uploaded only when all preceding required steps succeed.

### Why training and testing share one job

The tests need the newly trained model. Separate jobs would not have access to the first job's generated model unless it were explicitly transferred. A single job is simpler and guarantees the tests run against the model produced in that same run.

### Why `model_package/` is not committed

`model_package/` is listed in `.gitignore` on purpose. The model is generated by the pipeline and published as an artifact, which keeps **source code** separate from the **generated model artifact**.

---

## 16. Failure and Recovery Demonstrations

Two deliberate failures were demonstrated to verify that no model package is published when an important validation step fails. The following stayed unchanged throughout:

- Metric: F1-score
- Validation split: 80/20
- Random state: 42
- Improvement margin: 0.05

### Failure A: Quality Gate Failure

**Change:** the Random Forest candidate was temporarily replaced with `DummyClassifier(strategy="most_frequent")`, the same strategy as the baseline, so it could not demonstrate improvement. The metric, split, random state, margin, and gate condition were not modified.

**Result:**

```text
Baseline F1: 0.0000
Model F1:    0.0000
Margin:      0.0500
Required F1: 0.0500
QUALITY GATE FAILED.

0.0000 < 0.0500
```

`train.py` exited with a non-zero status. The workflow stopped at **Train model and run quality gate**, later steps did not run, and the model package was **not** published.

**Evidence:** [Failure A – Quality Gate Failure](https://github.com/saniyasawal/hypertension-risk/actions)

### Failure B: Application Test Failure

**Change:** in `src/predict.py`, the prediction function was temporarily changed from `return predictions` (an array) to `return int(predictions[0])` (a scalar), violating the expected output format.

**Result:**

```text
test_model_can_be_loaded                  PASSED
test_valid_sample_prediction              FAILED
test_missing_required_feature_is_rejected PASSED

AttributeError: 'int' object has no attribute 'shape'
1 failed, 2 passed
```

Training and the quality gate passed, but the application test stage failed. The workflow stopped before artifact publication, so no model package was uploaded.

**Evidence:** [Failure B – Application Test Failure](https://github.com/saniyasawal/hypertension-risk/actions)

### Recovery

After both failures, the original configuration was restored:

- The Random Forest candidate (`n_estimators=100`, `random_state=42`, `class_weight="balanced"`, `n_jobs=-1`).
- The prediction function (`return predictions`).

The F1 metric, 80/20 split, random state, and 0.05 margin were unchanged. The local training pipeline was re-run and the quality gate passed; the test suite was re-run and all three tests passed.

---

## 17. Final Successful Run

```text
Dataset validation        PASSED
Model training            PASSED
Quality gate              PASSED
Application tests         PASSED
Artifact creation         PASSED
Artifact upload           PASSED
```

### Evidence Links

| Run | Link |
|---|---|
| Initial successful run | [Initial Successful Run](Zip file in project folder) |
| Failure A: Quality gate failure | [Failure A](https://github.com/saniyasawal/hypertension-risk/actions) |
| Failure B: Application test failure | [Failure B](https://github.com/saniyasawal/hypertension-risk/actions) |
| Final successful run | [Final Successful Run](https://github.com/saniyasawal/hypertension-risk/actions) |

### Final Artifact

```text
hypertension-model-run-<FINAL_RUN_NUMBER>
```

Contents: `model.joblib`, `metrics.json`, `predict.py`, `requirements.txt`.

The artifact was generated only after the final training, quality gate, and application tests all passed. A copy was downloaded and retained as evidence.

---

## 18. Failure Conditions

The pipeline is designed to fail when:

| Condition | Behavior |
|---|---|
| Dataset file missing | `Dataset not found`; training fails |
| Required columns missing | Dataset validation fails |
| Dataset empty | Validation fails |
| `Model F1 < Baseline F1 + 0.05` | Quality gate fails |
| Any Pytest test fails | Workflow fails before artifact publication |

Required columns: `Gender`, `age`, `currentSmoker`, `cigsPerDay`, `BPMeds`, `diabetes`, `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`, `Risk`.

---

## 19. Reproducibility

The pipeline is reproducible through these fixed settings:

- Train/validation split: `test_size=0.20`, `random_state=42`, `stratify=y`
- Random Forest: `random_state=42`
- Quality margin: `0.05`
- Dataset stored directly in the repository
- Dependencies listed in `requirements.txt`
- Training and testing run as Python scripts, not notebooks

The GitHub Actions runner can therefore reproduce training and testing independently of the developer's machine.

---

## 20. How to Run Locally

```bash
# 1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hypertension-risk

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (should print "QUALITY GATE PASSED.")
python src/train.py

# 5. Run the prediction script on the sample input
python src/predict.py

# 6. Run automated tests (should report "3 passed")
pytest -v
```

Training saves the model to `model_package/model.joblib` and metrics to `model_package/metrics.json`.

---

## 21. MLOps Maturity Level

This project demonstrates an **early-stage / foundational MLOps workflow**, including:

- Version control with Git and a GitHub remote repository
- Automated CI with GitHub Actions
- Automated data validation
- Reproducible train/validation split
- Baseline comparison
- Automated model quality gate
- Automated application testing
- Model packaging and artifact publishing
- Failure detection and recovery demonstration

It is not a complete production MLOps system.

### Limitations

The pipeline does not yet include:

- Model monitoring, performance monitoring, or data drift detection
- Model registry or model version management beyond workflow artifacts
- Production deployment, infrastructure, or containerization
- Automated rollback or continuous retraining
- Feature store or experiment tracking
- Security scanning

It is primarily focused on CI for ML training, validation, testing, and artifact delivery.

### Next MLOps Level

```text
Data ingestion → Data validation → Feature engineering → Model training
→ Experiment tracking → Model evaluation → Model registry → Approval gate
→ Deployment → Monitoring → Drift detection → Automated retraining
```
MLOps maturity level: This implementation represents a basic/early MLOps level because it provides automated validation, reproducible training, quality gates, automated testing, continuous integration, and artifact delivery. However, it does not yet provide production deployment, model monitoring, drift detection, model registry, or automated retraining. These capabilities would be required to move toward the next level of MLOps maturity.

Possible future improvements:

- Add experiment tracking and a model registry
- Containerize the application and deploy the model as an API
- Add data drift detection and production performance monitoring
- Automate retraining when performance or data distribution changes
- Add security and dependency scanning
- Introduce model version management and automated deployment after validation

---

## 22. Assignment Requirements Checklist

| Requirement | Status |
|---|---|
| Tabular classification dataset | Completed |
| Dataset below 5 MB | Completed |
| Dataset stored in repository | Completed |
| Target and features documented | Completed |
| Baseline model implemented | Completed |
| Candidate model implemented | Completed |
| Reproducible train/validation split | Completed |
| Preprocessing fitted only on training data | Completed |
| Evaluation metric selected and justified | Completed |
| Improvement margin defined | Completed |
| Quality gate implemented | Completed |
| Pipeline fails when quality gate fails | Completed |
| Automated application tests | Completed |
| Model loading test | Completed |
| Valid prediction test | Completed |
| Missing-feature validation test | Completed |
| GitHub Actions workflow | Completed |
| Training and tests in same job | Completed |
| Model package creation | Completed |
| Artifact upload | Completed |
| Failure A demonstrated | Completed |
| Failure B demonstrated | Completed |
| Recovery demonstrated | Completed |
| Final successful run | Completed |
| Artifact identified by workflow run | Completed |
| README documentation | Completed |

---

## 23. Conclusion

This project demonstrates an automated ML pipeline for hypertension risk classification using GitHub Actions. Instead of manually training and testing the model, the workflow validates the dataset, trains a baseline and a candidate model, evaluates the candidate with F1-score, applies a predefined quality gate, runs application tests, and publishes the model package only when all required checks pass.

Two deliberate failures were demonstrated. The first showed that a model failing the required improvement margin triggers the quality gate and blocks artifact publication. The second showed that even when the model passes the quality gate, an application-level test failure still prevents publication. After both failures, the original configuration was restored and the complete pipeline ran successfully.

The result is a foundational MLOps process with automated validation, reproducible training, quality control, application testing, and artifact delivery.