"""Feature definitions and target preparation."""

import pandas as pd


TARGET_COLUMN = "num"

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def prepare_target(data: pd.DataFrame) -> pd.Series:
    """
    Convert the original UCI target into a binary target.

    Original target:
        0 = absence of heart disease
        >0 = presence of heart disease

    Returns
    -------
    pandas.Series
        Binary target containing 0 or 1.
    """
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    return (data[TARGET_COLUMN] > 0).astype(int)


def split_features_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate input features and binary target.
    """
    missing_features = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]

    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    X = data[FEATURE_COLUMNS].copy()
    y = prepare_target(data)

    return X, y