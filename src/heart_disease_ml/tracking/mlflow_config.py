"""MLflow configuration utilities."""

import mlflow

from heart_disease_ml.utils.config import load_mlflow_config


def configure_mlflow() -> str:
    """Configure MLflow tracking URI and experiment."""

    config = load_mlflow_config()["mlflow"]

    tracking_uri = config["tracking_uri"]
    experiment_name = config["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    return experiment_name