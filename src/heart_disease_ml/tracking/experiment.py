"""Reusable MLflow experiment logging utilities."""

from pathlib import Path
from typing import Any

import mlflow


def start_experiment_run(
    run_name: str,
    tags: dict[str, str] | None = None,
):
    """Start an MLflow run."""

    return mlflow.start_run(
        run_name=run_name,
        tags=tags,
    )


def log_parameters(parameters: dict[str, Any]) -> None:
    """Log experiment parameters."""

    mlflow.log_params(
        {
            key: str(value)
            for key, value in parameters.items()
        }
    )


def log_metrics(metrics: dict[str, float]) -> None:
    """Log experiment metrics."""

    cleaned_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if value is not None
    }

    mlflow.log_metrics(cleaned_metrics)


def log_artifacts(
    artifacts: dict[str, str | Path],
) -> None:
    """Log generated artifact files."""

    for artifact_path in artifacts.values():
        mlflow.log_artifact(str(artifact_path))


def log_model(
    model: Any,
    artifact_path: str = "model",
) -> None:
    """Log a scikit-learn model to MLflow."""

    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
    )


def set_experiment_tags(tags: dict[str, str]) -> None:
    """Set tags on the active MLflow run."""

    mlflow.set_tags(tags)