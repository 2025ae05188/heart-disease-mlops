"""Reusable sklearn preprocessing pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_disease_ml.preprocessing.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def _categorical_transformer() -> Pipeline:
    """Create the categorical preprocessing pipeline."""
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )


def _numerical_transformer(scale: bool = False) -> Pipeline:
    """Create the numerical preprocessing pipeline."""
    steps = [
        (
            "imputer",
            SimpleImputer(strategy="median"),
        )
    ]

    if scale:
        steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    return Pipeline(steps=steps)


def create_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """
    Create the common preprocessing transformer.

    Parameters
    ----------
    scale_numeric:
        Whether numerical features should be standardized.

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                _numerical_transformer(scale=scale_numeric),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                _categorical_transformer(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def create_logistic_preprocessor() -> ColumnTransformer:
    """Preprocessor used by Logistic Regression."""
    return create_preprocessor(scale_numeric=True)


def create_random_forest_preprocessor() -> ColumnTransformer:
    """Preprocessor used by Random Forest."""
    return create_preprocessor(scale_numeric=False)