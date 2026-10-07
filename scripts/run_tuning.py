"""Run and track hyperparameter tuning experiments with MLflow."""

import json
from tempfile import TemporaryDirectory

import mlflow
import mlflow.sklearn
from mlflow.models import ModelSignature
from mlflow.types import ColSpec, Schema
from sklearn.model_selection import ParameterGrid, train_test_split

from heart_disease_ml.data.load import load_csv
from heart_disease_ml.evaluation.metrics import evaluate_model
from heart_disease_ml.evaluation.reports import generate_evaluation_artifacts
from heart_disease_ml.models.factory import create_model_pipeline
from heart_disease_ml.models.tune import (
    get_default_parameter_grids,
    tune_model,
)
from heart_disease_ml.preprocessing.cleaning import clean_heart_disease_data
from heart_disease_ml.preprocessing.features import split_features_target
from heart_disease_ml.tracking.mlflow_config import configure_mlflow
from heart_disease_ml.utils.config import (
    load_data_config,
    load_model_config,
    load_project_config,
)
from heart_disease_ml.utils.fingerprint import fingerprint_file
from heart_disease_ml.utils.paths import RAW_DATA_DIR


def create_model_signature() -> ModelSignature:
    """Create an explicit MLflow input signature for raw feature data."""

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

    return ModelSignature(inputs=input_schema)


def log_cv_candidates(
    search,
    model_name: str,
    dataset_name: str,
    dataset_fingerprint: str,
    cv_folds: int,
) -> None:
    """Log every GridSearchCV candidate as an MLflow nested run."""

    results = search.cv_results_

    for index, params in enumerate(results["params"]):
        candidate_run_name = (
            f"tuning_grid_search_{model_name}_candidate_{index + 1:02d}"
        )

        with mlflow.start_run(
            run_name=candidate_run_name,
            nested=True,
            tags={
                "assignment_section": "3",
                "experiment_type": "tuning_candidate",
                "model_family": model_name,
                "tuning_method": "grid_search",
                "run_role": "candidate",
                "dataset_name": dataset_name,
                "dataset_fingerprint": dataset_fingerprint,
                "selection_metric": "cv_roc_auc_mean",
            },
        ):
            # Log candidate hyperparameters.
            clean_params = {
                key.replace("model__", ""): value
                for key, value in params.items()
            }

            mlflow.log_params(clean_params)

            # Log common CV configuration.
            mlflow.log_params(
                {
                    "cv_folds": cv_folds,
                    "cv_scoring": "roc_auc",
                }
            )

            # Log candidate CV performance.
            mlflow.log_metric(
                "cv_roc_auc_mean",
                float(results["mean_test_score"][index]),
            )

            mlflow.log_metric(
                "cv_roc_auc_std",
                float(results["std_test_score"][index]),
            )

            # Log timing information.
            mlflow.log_metric(
                "mean_fit_time",
                float(results["mean_fit_time"][index]),
            )

            mlflow.log_metric(
                "mean_score_time",
                float(results["mean_score_time"][index]),
            )


def run_tuning(model_name: str) -> None:
    """Run GridSearchCV and track the tuning process in MLflow."""

    # ------------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------------

    project_config = load_project_config()
    data_config = load_data_config()
    model_config = load_model_config()

    random_state = project_config["project"]["random_state"]

    test_size = model_config["training"]["test_size"]
    cv_folds = model_config["training"]["cv_folds"]
    scoring = model_config["training"]["scoring"]

    dataset_name = data_config["dataset"]["name"]
    dataset_id = data_config["dataset"]["dataset_id"]

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------

    model = create_model_pipeline(model_name)

    # ------------------------------------------------------------------
    # Select tuning grid
    #
    # YAML configuration takes priority.
    # Python defaults are used only when YAML does not provide a grid.
    # ------------------------------------------------------------------

    default_parameter_grids = get_default_parameter_grids()

    yaml_tuning_config = model_config.get("tuning", {})
    yaml_model_config = yaml_tuning_config.get(model_name, {})
    yaml_parameters = yaml_model_config.get("parameters")

    if yaml_parameters:
        param_grid = yaml_parameters
        tuning_grid_source = "yaml"
    else:
        param_grid = default_parameter_grids[model_name]
        tuning_grid_source = "default"

    candidate_count = len(list(ParameterGrid(param_grid)))

    parent_run_name = f"tuning_grid_search_{model_name}"

    # ------------------------------------------------------------------
    # Parent MLflow run
    # ------------------------------------------------------------------

    with mlflow.start_run(
        run_name=parent_run_name,
        tags={
            "assignment_section": "3",
            "tuning_grid_source": tuning_grid_source,
            "experiment_type": "hyperparameter_tuning",
            "model_family": model_name,
            "tuning_method": "grid_search",
            "run_role": "tuning_parent",
            "dataset_name": dataset_name,
            "dataset_id": str(dataset_id),
            "dataset_fingerprint": dataset_fingerprint,
            "selection_metric": "cv_roc_auc_mean",
            "preprocessing": "sklearn_pipeline",
        },
        description=(
            f"GridSearchCV hyperparameter tuning for {model_name} "
            "using training data only."
        ),
    ):
        # --------------------------------------------------------------
        # Log tuning configuration
        # --------------------------------------------------------------

        mlflow.log_param(
            "tuning_grid_source",
            tuning_grid_source,
        )

        mlflow.log_params(
            {
                "random_state": random_state,
                "test_size": test_size,
                "cv_folds": cv_folds,
                "cv_scoring": scoring,
                "dataset_rows": len(data),
                "dataset_features": X.shape[1],
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "candidate_count": candidate_count,
            }
        )

        # Store the exact parameter grid used for this experiment.
        mlflow.log_text(
            json.dumps(param_grid, indent=2, default=str),
            "tuning/parameter_grid.json",
        )

        # --------------------------------------------------------------
        # Display tuning configuration
        # --------------------------------------------------------------

        print(f"\nTuning model: {model_name}")
        print(f"Parameter grid source: {tuning_grid_source}")
        print(f"Parameter grid: {param_grid}")
        print(f"Number of candidates: {candidate_count}")
        print(f"CV folds: {cv_folds}")
        print(f"Scoring: {scoring}")
        print("\nRunning GridSearchCV...\n")

        # --------------------------------------------------------------
        # Perform GridSearchCV
        # --------------------------------------------------------------

        search = tune_model(
            model=model,
            param_grid=param_grid,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            cv_splits=cv_folds,
            n_jobs=-1,
        )

        # --------------------------------------------------------------
        # Best configuration selected using CV
        # --------------------------------------------------------------

        print("Best parameters:")
        print(search.best_params_)

        print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")

        mlflow.log_metric(
            "best_cv_roc_auc",
            float(search.best_score_),
        )

        for key, value in search.best_params_.items():
            clean_key = key.replace("model__", "best_")
            mlflow.log_param(clean_key, value)

        # --------------------------------------------------------------
        # Log all candidate configurations
        # --------------------------------------------------------------

        log_cv_candidates(
            search=search,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_fingerprint=dataset_fingerprint,
            cv_folds=cv_folds,
        )

        # --------------------------------------------------------------
        # Final test evaluation
        #
        # IMPORTANT:
        # The test set was not used during GridSearchCV.
        # It is evaluated only once after selecting the best CV model.
        # --------------------------------------------------------------

        best_model = search.best_estimator_

        test_metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
        )

        print("\nFinal test metrics:")

        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

            mlflow.log_metric(
                f"test_{metric}",
                float(value),
            )

        # --------------------------------------------------------------
        # Evaluation artifacts
        # --------------------------------------------------------------

        with TemporaryDirectory() as temp_dir:
            generate_evaluation_artifacts(
                best_model,
                X_test,
                y_test,
                output_dir=temp_dir,
                model_name=f"tuned_{model_name}",
            )

            mlflow.log_artifacts(
                temp_dir,
                artifact_path="evaluation",
            )

        # --------------------------------------------------------------
        # Log best tuned model
        # --------------------------------------------------------------

        signature = create_model_signature()

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3),
        )

        print("\nTuning completed successfully.")
        print(f"Model: {model_name}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")


def main() -> None:
    """Run tuning for all enabled models."""

    configure_mlflow()

    model_config = load_model_config()

    enabled_models = [
        model_name
        for model_name, config in model_config["models"].items()
        if config.get("enabled", False)
    ]

    print("Enabled models:")

    for model_name in enabled_models:
        print(f"  - {model_name}")

    for model_name in enabled_models:
        run_tuning(model_name)


if __name__ == "__main__":
    main()