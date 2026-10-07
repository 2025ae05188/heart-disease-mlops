"""Verify that the standalone champion model works without MLflow."""

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "champion" / "model.pkl"

SAMPLE_INPUT = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}


def main() -> None:
    """Load the standalone model and perform one prediction."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Champion model not found: {MODEL_PATH}"
        )

    print(f"Loading model: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([SAMPLE_INPUT])

    prediction = model.predict(sample)[0]
    probabilities = model.predict_proba(sample)[0]

    confidence = float(probabilities[prediction])

    print("\nModel verification successful.")
    print("----------------------------------------")
    print(f"Model type : {type(model).__name__}")
    print(f"Prediction : {int(prediction)}")
    print(f"Confidence : {confidence:.4f}")


if __name__ == "__main__":
    main()