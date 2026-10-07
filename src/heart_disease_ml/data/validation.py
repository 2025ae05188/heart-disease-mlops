"""Dataset validation utilities."""

from collections.abc import Sequence

import pandas as pd


EXPECTED_COLUMNS = [
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
    "num",
]


def validate_columns(
    data: pd.DataFrame,
    expected_columns: Sequence[str] = EXPECTED_COLUMNS,
) -> None:
    """
    Validate that the dataset contains the expected columns.

    Raises
    ------
    ValueError
        If columns are missing or unexpected columns are present.
    """
    actual_columns = list(data.columns)
    expected_columns = list(expected_columns)

    missing = sorted(set(expected_columns) - set(actual_columns))
    unexpected = sorted(set(actual_columns) - set(expected_columns))

    if missing or unexpected:
        raise ValueError(
            f"Dataset columns do not match expected schema. "
            f"Missing: {missing}; Unexpected: {unexpected}"
        )


def validate_not_empty(data: pd.DataFrame) -> None:
    """
    Validate that the dataset contains at least one row.
    """
    if data.empty:
        raise ValueError("Dataset is empty.")


def validate_heart_disease_data(data: pd.DataFrame) -> None:
    """
    Run the basic validation checks for the UCI Heart Disease dataset.
    """
    validate_not_empty(data)
    validate_columns(data)