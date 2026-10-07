"""Tests for model evaluation metrics."""

import numpy as np

from heart_disease_ml.evaluation.metrics import evaluate_model
from heart_disease_ml.models.factory import create_model_pipeline


def test_evaluate_model_returns_expected_metrics():
    """Model evaluation should return the required classification metrics."""

    X = np.array(
        [
            [50, 1, 1, 120, 200, 0, 0, 150, 0, 1.0, 1, 0, 1],
            [60, 0, 2, 140, 240, 0, 1, 130, 1, 2.0, 2, 1, 2],
            [45, 1, 0, 110, 180, 0, 0, 170, 0, 0.5, 1, 0, 1],
            [65, 0, 3, 150, 260, 1, 2, 120, 1, 2.5, 2, 2, 3],
            [55, 1, 2, 130, 220, 0, 1, 140, 0, 1.5, 1, 0, 2],
            [40, 0, 1, 115, 190, 0, 0, 175, 0, 0.2, 1, 0, 1],
        ]
    )

    y = np.array([0, 1, 0, 1, 1, 0])

    # Use the project's actual feature names.
    feature_names = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    import pandas as pd

    X = pd.DataFrame(X, columns=feature_names)

    model = create_model_pipeline("logistic_regression")
    model.fit(X, y)

    metrics = evaluate_model(model, X, y)

    expected_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert set(metrics.keys()) == expected_metrics

    for value in metrics.values():
        assert 0.0 <= value <= 1.0