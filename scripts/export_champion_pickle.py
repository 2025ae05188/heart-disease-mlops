"""Export the packaged champion MLflow model as a pickle file."""

import joblib
import mlflow

from heart_disease_ml.tracking.mlflow_config import configure_mlflow
from heart_disease_ml.utils.paths import MODELS_DIR


def main() -> None:
    """Load the packaged champion model and export it as a pickle."""

    configure_mlflow()

    champion_dir = MODELS_DIR / "champion"
    mlflow_model_dir = champion_dir / "model"
    pickle_path = champion_dir / "model.pkl"

    if not mlflow_model_dir.exists():
        raise FileNotFoundError(
            f"Packaged MLflow model not found: {mlflow_model_dir}"
        )

    print(f"Loading packaged model from: {mlflow_model_dir}")

    model = mlflow.sklearn.load_model(str(mlflow_model_dir))

    joblib.dump(model, pickle_path)

    print("\nPickle export completed.")
    print("----------------------------------------")
    print(f"Model type : {type(model).__name__}")
    print(f"Pickle     : {pickle_path}")
    print(f"Size       : {pickle_path.stat().st_size / 1024:.2f} KB")


if __name__ == "__main__":
    main()