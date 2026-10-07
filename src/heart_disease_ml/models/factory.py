"""Factory functions for constructing complete ML pipelines."""

from sklearn.pipeline import Pipeline

from heart_disease_ml.models.definitions import (
    create_logistic_regression,
    create_random_forest,
)
from heart_disease_ml.preprocessing.transformers import (
    create_logistic_preprocessor,
    create_random_forest_preprocessor,
)


def create_logistic_pipeline(**model_params) -> Pipeline:
    """
    Create the complete Logistic Regression pipeline.

    Pipeline:
        preprocessing -> Logistic Regression
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_logistic_preprocessor(),
            ),
            (
                "model",
                create_logistic_regression(**model_params),
            ),
        ]
    )


def create_random_forest_pipeline(**model_params) -> Pipeline:
    """
    Create the complete Random Forest pipeline.

    Pipeline:
        preprocessing -> Random Forest
    """
    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_random_forest_preprocessor(),
            ),
            (
                "model",
                create_random_forest(**model_params),
            ),
        ]
    )


def create_model_pipeline(
    model_name: str,
    **model_params,
) -> Pipeline:
    """
    Create a model pipeline by model name.
    """
    model_name = model_name.lower()

    if model_name in {"logistic_regression", "logistic regression", "logistic", "lr"}:
        return create_logistic_pipeline(**model_params)

    if model_name in {"random_forest", "random forest", "randomforest", "rf"}:
        return create_random_forest_pipeline(**model_params)

    raise ValueError(
        f"Unsupported model '{model_name}'. "
        "Supported models: logistic_regression, random_forest."
    )