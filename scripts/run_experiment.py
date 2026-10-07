"""Run and track baseline heart disease classification experiments."""

from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec
from sklearn.model_selection import train_test_split

from heart_disease_ml.data.load import load_csv
from heart_disease_ml.preprocessing.cleaning import clean_heart_disease_data
from heart_disease_ml.preprocessing.features import split_features_target
from heart_disease_ml.models.factory import create_model_pipeline
from heart_disease_ml.models.train import train_model
from heart_disease_ml.evaluation.cross_validation import cross_validate_model
from heart_disease_ml.evaluation.metrics import evaluate_model
from heart_disease_ml.evaluation.reports import generate_evaluation_artifacts
from heart_disease_ml.tracking.mlflow_config import configure_mlflow
from heart_disease_ml.utils.config import (
    load_data_config,
    load_model_config,
    load_project_config,
)
from heart_disease_ml.utils.fingerprint import fingerprint_file
from heart_disease_ml.utils.paths import RAW_DATA_DIR


def run_experiment(model_name: str) -> None:
    """Run one baseline model experiment and track it with MLflow."""

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------
    project_config = load_project_config()
    data_config = load_data_config()
    model_config = load_model_config()

    random_state = project_config["project"]["random_state"]
    test_size = model_config["training"]["test_size"]
    cv_folds = model_config["training"]["cv_folds"]
    scoring = model_config["training"]["scoring"]

    dataset_name = data_config["dataset"]["name"]
    dataset_id = data_config["dataset"]["dataset_id"]

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    data_path = RAW_DATA_DIR / "heart_disease.csv"

    data = load_csv(data_path)
    data = clean_heart_disease_data(data)

    X, y = split_features_target(data)

    dataset_fingerprint = fingerprint_file(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------------------------------------------------------
    # MLflow run
    # ---------------------------------------------------------
    run_name = f"baseline__{model_name}"

    tags = {
        "assignment_section": "3",
        "experiment_type": "baseline",
        "model_family": model_name,
        "dataset_name": dataset_name,
        "dataset_id": str(dataset_id),
        "dataset_fingerprint": dataset_fingerprint,
        "run_role": "baseline",
        "selection_metric": "cv_roc_auc_mean",
        "tuning_method": "none",
        "preprocessing": "sklearn_pipeline",
    }

    with mlflow.start_run(
        run_name=run_name,
        tags=tags,
        description=(
            f"Baseline {model_name} experiment for the UCI Heart Disease "
            "classification assignment."
        ),
    ):

        # -----------------------------------------------------
        # Log parameters
        # -----------------------------------------------------
        mlflow.log_params(
            {
                "model": model_name,
                "random_state": random_state,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "cv_scoring": scoring,
                "dataset_rows": len(data),
                "dataset_features": X.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            }
        )

        # -----------------------------------------------------
        # Train
        # -----------------------------------------------------
        model = create_model_pipeline(model_name)

        trained_model = train_model(
            model,
            X_train,
            y_train,
        )

        # -----------------------------------------------------
        # Cross-validation
        # -----------------------------------------------------
        cv_results = cross_validate_model(
            trained_model,
            X_train,
            y_train,
            n_splits=cv_folds,
        )

        # Log CV summary metrics.
        for metric, value in cv_results.items():
            if metric == "cv_results":
                continue

            mlflow.log_metric(
                metric,
                float(value),
            )

        # -----------------------------------------------------
        # Test evaluation
        # -----------------------------------------------------
        test_metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
        )

        for metric, value in test_metrics.items():
            mlflow.log_metric(
                f"test_{metric}",
                float(value),
            )

        # -----------------------------------------------------
        # Evaluation artifacts
        # -----------------------------------------------------
        with TemporaryDirectory() as temp_dir:
            generate_evaluation_artifacts(
                trained_model,
                X_test,
                y_test,
                output_dir=temp_dir,
                model_name=model_name,
            )

            mlflow.log_artifacts(
                temp_dir,
                artifact_path="evaluation",
            )

        # -----------------------------------------------------
        # Log model
        # -----------------------------------------------------
        input_schema = Schema(
            [
        ColSpec("double", "age"),
        ColSpec("long", "sex"),
        ColSpec("long", "cp"),
        ColSpec("double", "trestbps"),
        ColSpec("double", "chol"),
        ColSpec("long", "fbs"),
        ColSpec("long", "restecg"),
        ColSpec("double", "thalach"),
        ColSpec("long", "exang"),
        ColSpec("double", "oldpeak"),
        ColSpec("long", "slope"),
        ColSpec("double", "ca"),
        ColSpec("double", "thal"),
            ]
        )

        signature = ModelSignature(inputs=input_schema)

        mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
        )

        # -----------------------------------------------------
        # Run information
        # -----------------------------------------------------
        run = mlflow.active_run()

        print("\n" + "=" * 70)
        print(f"MLflow run completed: {model_name}")
        print("=" * 70)

        print(f"Run name     : {run_name}")
        print(f"Run ID       : {run.info.run_id}")
        print(f"Experiment   : {run.info.experiment_id}")
        print(f"Dataset hash : {dataset_fingerprint}")

        print("\nTest metrics:")

        for metric, value in test_metrics.items():
            print(f"  {metric:10s}: {value:.4f}")


def main() -> None:
    """Run baseline experiments for all enabled models."""

    experiment_name = configure_mlflow()

    print(f"MLflow experiment: {experiment_name}")

    model_config = load_model_config()

    enabled_models = [
        model_name
        for model_name, config in model_config["models"].items()
        if config.get("enabled", False)
    ]

    print(f"Enabled models: {enabled_models}")

    for model_name in enabled_models:
        run_experiment(model_name)


if __name__ == "__main__":
    main()