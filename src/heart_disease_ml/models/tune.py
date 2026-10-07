"""Hyperparameter tuning utilities."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, StratifiedKFold


RANDOM_STATE = 42


def tune_model(
    model: BaseEstimator,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "roc_auc",
    cv_splits: int = 5,
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Perform hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    model:
        Complete sklearn pipeline to tune.
    param_grid:
        Hyperparameter combinations.
    X_train:
        Training features.
    y_train:
        Training target.
    scoring:
        Metric used to select the best configuration.
    cv_splits:
        Number of stratified cross-validation folds.
    n_jobs:
        Number of parallel jobs.

    Returns
    -------
    GridSearchCV
        Fitted GridSearchCV object.
    """
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
    )

    search.fit(X_train, y_train)

    return search


def get_default_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """
    Return baseline tuning grids for the supported models.
    """
    return {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
    }