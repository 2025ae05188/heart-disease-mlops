"""Train and evaluate a fresh model for CI validation.

This script is used only by the CI/CD pipeline.

It does not load, modify, register, or promote the production
champion model. Instead, it downloads the dataset, applies the
existing preprocessing pipeline, trains a fresh Logistic Regression
model using the selected hyperparameter C=2.0, evaluates it, and
stores CI artifacts.
"""

import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from heart_disease_ml.data.download import download_heart_disease
from heart_disease_ml.models.factory import create_model_pipeline
from heart_disease_ml.models.train import train_model
from heart_disease_ml.preprocessing.cleaning import clean_heart_disease_data
from heart_disease_ml.preprocessing.features import split_features_target
from heart_disease_ml.evaluation.metrics import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CI_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "ci"
CI_MODEL_PATH = CI_ARTIFACT_DIR / "ci_model.pkl"
CI_METRICS_PATH = CI_ARTIFACT_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
LOGISTIC_REGRESSION_C = 2.0


def main() -> None:
    """Run a fresh model training and evaluation for CI."""

    CI_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download the dataset using the existing project function.
    # ------------------------------------------------------------------
    print("Downloading UCI Heart Disease dataset...")

    data = download_heart_disease()

    print(f"Dataset shape: {data.shape}")

    # ------------------------------------------------------------------
    # 2. Clean the dataset using the existing preprocessing function.
    # ------------------------------------------------------------------
    print("Cleaning dataset...")

    data = clean_heart_disease_data(data)

    # ------------------------------------------------------------------
    # 3. Prepare features and target using the existing function.
    # ------------------------------------------------------------------
    X, y = split_features_target(data)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape : {y.shape}")

    # ------------------------------------------------------------------
    # 4. Create a fresh train/test split.
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")

    # ------------------------------------------------------------------
    # 5. Create the selected model using the existing model factory.
    # ------------------------------------------------------------------
    print("\nCreating Logistic Regression pipeline...")
    print(f"  C           : {LOGISTIC_REGRESSION_C}")
    print(f"  random_state: {RANDOM_STATE}")

    model = create_model_pipeline(
        "logistic_regression",
        C=LOGISTIC_REGRESSION_C,
    )

    # ------------------------------------------------------------------
    # 6. Train a fresh model using the existing training function.
    # ------------------------------------------------------------------
    print("\nTraining fresh model...")

    model = train_model(
        model,
        X_train,
        y_train,
    )

    # ------------------------------------------------------------------
    # 7. Evaluate using the existing project evaluation function.
    # ------------------------------------------------------------------
    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # ------------------------------------------------------------------
    # 8. Validate that all required metrics were produced.
    # ------------------------------------------------------------------
    required_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    for metric_name in required_metrics:
        if metric_name not in metrics:
            raise RuntimeError(
                f"Required metric missing: {metric_name}"
            )

        value = float(metrics[metric_name])

        if not 0.0 <= value <= 1.0:
            raise RuntimeError(
                f"Invalid {metric_name} value: {value}"
            )

    # ------------------------------------------------------------------
    # 9. Save the freshly trained CI model.
    #
    # This is NOT the production champion model.
    # ------------------------------------------------------------------
    joblib.dump(model, CI_MODEL_PATH)

    # ------------------------------------------------------------------
    # 10. Save evaluation metrics.
    # ------------------------------------------------------------------
    with CI_METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # ------------------------------------------------------------------
    # 11. Print a clear CI summary.
    # ------------------------------------------------------------------
    print("\nCI training completed successfully.")
    print("----------------------------------------")
    print("Model            : Logistic Regression")
    print(f"Hyperparameter C : {LOGISTIC_REGRESSION_C}")
    print(f"Model artifact   : {CI_MODEL_PATH}")
    print(f"Metrics artifact : {CI_METRICS_PATH}")

    print("\nEvaluation metrics:")

    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")


if __name__ == "__main__":
    main()