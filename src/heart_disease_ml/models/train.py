"""Model training utilities."""

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> BaseEstimator:
    """
    Fit a model pipeline on the training data.

    Parameters
    ----------
    model:
        An sklearn estimator or pipeline.
    X_train:
        Training features.
    y_train:
        Training target.

    Returns
    -------
    BaseEstimator
        The fitted model.
    """
    model.fit(X_train, y_train)

    return model


def train_model_by_name(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **model_params: Any,
) -> BaseEstimator:
    """
    Construct and train a model using its registered model name.
    """
    from heart_disease_ml.models.factory import create_model_pipeline

    model = create_model_pipeline(
        model_name,
        **model_params,
    )

    return train_model(
        model,
        X_train,
        y_train,
    )