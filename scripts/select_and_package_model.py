"""Select, register, and package the best MLflow tuning candidate."""

import json
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from heart_disease_ml.tracking.mlflow_config import configure_mlflow
from heart_disease_ml.utils.paths import MODELS_DIR


def get_best_tuning_candidate(
    client: MlflowClient,
    experiment_name: str,
):
    """Return the tuning candidate with the highest CV ROC-AUC."""

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {experiment_name}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.experiment_type = 'tuning_candidate'",
        order_by=["metrics.cv_roc_auc_mean DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError(
            "No tuning candidate runs were found in the MLflow experiment."
        )

    return runs[0]


def get_tuning_parent(
    client: MlflowClient,
    experiment_name: str,
    model_family: str,
):
    """Find the tuning parent for the selected model family."""

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment not found: {experiment_name}"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "tags.experiment_type = 'hyperparameter_tuning' "
            f"and tags.model_family = '{model_family}'"
        ),
        max_results=10,
    )

    if not runs:
        raise RuntimeError(
            f"No tuning parent found for model family: {model_family}"
        )

    if len(runs) > 1:
        raise RuntimeError(
            f"Expected one tuning parent for {model_family}, "
            f"but found {len(runs)}."
        )

    return runs[0]


def register_champion_model(
    client: MlflowClient,
    model_name: str,
    parent_run_id: str,
):
    """Register the existing trained model and assign the champion alias."""

    model_uri = f"runs:/{parent_run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered_model.version,
    )

    return registered_model


def clean_previous_package(output_dir: Path) -> None:
    """Remove an existing champion package."""

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def package_model(
    parent_run_id: str,
    candidate_run,
    output_dir: Path,
) -> None:
    """Download the trained model artifact from the tuning parent."""

    downloaded_model = mlflow.artifacts.download_artifacts(
        run_id=parent_run_id,
        artifact_path="model",
    )

    destination = output_dir / "model"

    shutil.copytree(downloaded_model, destination)

    selection_metadata = {
        "selection_method": "highest_cv_roc_auc_mean",
        "selection_metric": "cv_roc_auc_mean",
        "candidate_run_id": candidate_run.info.run_id,
        "candidate_run_name": candidate_run.data.tags.get("mlflow.runName"),
        "model_family": candidate_run.data.tags.get("model_family"),
        "cv_roc_auc_mean": candidate_run.data.metrics.get(
            "cv_roc_auc_mean"
        ),
        "cv_roc_auc_std": candidate_run.data.metrics.get(
            "cv_roc_auc_std"
        ),
        "dataset_name": candidate_run.data.tags.get("dataset_name"),
        "dataset_fingerprint": candidate_run.data.tags.get(
            "dataset_fingerprint"
        ),
        "parent_run_id": parent_run_id,
        "model_artifact_source": f"runs:/{parent_run_id}/model",
        "candidate_parameters": candidate_run.data.params,
    }

    metadata_path = output_dir / "selection.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(selection_metadata, file, indent=2, default=str)

    print("\nModel packaging completed.")
    print("----------------------------------------")
    print(f"Model family       : {selection_metadata['model_family']}")
    print(
        "CV ROC-AUC         : "
        f"{selection_metadata['cv_roc_auc_mean']:.4f}"
    )
    print(f"Candidate run ID   : {selection_metadata['candidate_run_id']}")
    print(f"Parent run ID      : {selection_metadata['parent_run_id']}")
    print(f"Model package      : {destination}")
    print(f"Selection metadata : {metadata_path}")


def main() -> None:
    """Select, register, and package the best tuned model."""

    experiment_name = configure_mlflow()
    client = MlflowClient()

    print(f"MLflow experiment: {experiment_name}")
    print("\nSearching existing tuning candidates...")

    candidate = get_best_tuning_candidate(
        client=client,
        experiment_name=experiment_name,
    )

    model_family = candidate.data.tags.get("model_family")

    if not model_family:
        raise RuntimeError(
            "Selected tuning candidate does not contain model_family tag."
        )

    print("\nSelected tuning candidate:")
    print(f"  Run name      : {candidate.data.tags.get('mlflow.runName')}")
    print(f"  Run ID        : {candidate.info.run_id}")
    print(f"  Model family  : {model_family}")
    print(
        "  CV ROC-AUC    : "
        f"{candidate.data.metrics['cv_roc_auc_mean']:.4f}"
    )
    print(f"  Parameters    : {candidate.data.params}")

    parent = get_tuning_parent(
        client=client,
        experiment_name=experiment_name,
        model_family=model_family,
    )

    print("\nCorresponding tuning parent:")
    print(f"  Run name : {parent.data.tags.get('mlflow.runName')}")
    print(f"  Run ID   : {parent.info.run_id}")

    # Register the already-trained model from the tuning parent.
    # This does NOT retrain or rerun hyperparameter tuning.
    registered_model = register_champion_model(
        client=client,
        model_name="heart-disease-classifier",
        parent_run_id=parent.info.run_id,
    )

    print("\nMLflow Model Registry:")
    print(f"  Model name : {registered_model.name}")
    print(f"  Version    : {registered_model.version}")
    print("  Alias      : @champion")

    output_dir = MODELS_DIR / "champion"

    clean_previous_package(output_dir)

    package_model(
        parent_run_id=parent.info.run_id,
        candidate_run=candidate,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()