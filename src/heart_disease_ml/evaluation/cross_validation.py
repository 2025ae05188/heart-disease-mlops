"""Cross-validation evaluation utilities."""

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate


RANDOM_STATE = 42

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.

    Parameters
    ----------
    model:
        Complete sklearn model pipeline.
    X:
        Training features.
    y:
        Training target.
    n_splits:
        Number of CV folds.
    n_jobs:
        Number of parallel jobs.

    Returns
    -------
    dict
        Fold-level scores and mean/std for each metric.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=n_jobs,
        return_train_score=False,
    )

    summary = {}

    for metric in CV_SCORING:
        scores = results[f"test_{metric}"]

        summary[f"cv_{metric}_mean"] = float(scores.mean())
        summary[f"cv_{metric}_std"] = float(scores.std())

    summary["cv_results"] = results

    return summary