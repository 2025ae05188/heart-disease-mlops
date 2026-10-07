"""Classification evaluation metrics."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_prob: Any,
) -> dict[str, float]:
    """
    Calculate the classification metrics used by the project.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Predicted binary labels.
    y_prob:
        Predicted probability for the positive class.

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, F1 and ROC-AUC.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
    }


def predict_for_evaluation(model: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate class predictions and positive-class probabilities.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return y_pred, y_prob


def evaluate_model(
    model: Any,
    X: Any,
    y: Any,
) -> dict[str, float]:
    """
    Generate predictions and calculate all project metrics.
    """
    y_pred, y_prob = predict_for_evaluation(model, X)

    return calculate_classification_metrics(
        y_true=y,
        y_pred=y_pred,
        y_prob=y_prob,
    )