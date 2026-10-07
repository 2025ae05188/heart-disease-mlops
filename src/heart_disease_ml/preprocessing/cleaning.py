"""Raw dataset cleaning utilities."""

import pandas as pd


def clean_heart_disease_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning of the raw UCI Heart Disease dataset.

    The UCI dataset represents some missing values using '?'.
    These values are converted to pandas missing values.

    Numeric columns are converted to numeric dtype where possible.
    Missing-value imputation is intentionally not performed here;
    it is handled by the sklearn preprocessing pipelines.
    """
    cleaned = data.copy()

    # Convert UCI missing-value markers to pandas NaN.
    cleaned = cleaned.replace("?", pd.NA)

    # Columns that should contain numeric values.
    numeric_columns = [
        "age",
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
        "num",
    ]

    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    return cleaned