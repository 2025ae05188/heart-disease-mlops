"""Tests for preprocessing utilities."""

import pandas as pd

from heart_disease_ml.preprocessing.cleaning import (
    clean_heart_disease_data,
)


def test_question_mark_values_are_converted_to_missing():
    """Question-mark values should be converted to missing values."""

    data = pd.DataFrame(
        {
            "age": [63],
            "ca": ["?"],
            "thal": ["?"],
        }
    )

    cleaned = clean_heart_disease_data(data)

    assert pd.isna(cleaned.loc[0, "ca"])
    assert pd.isna(cleaned.loc[0, "thal"])


def test_numeric_columns_are_converted():
    """Numeric feature values should be converted to numeric dtype."""

    data = pd.DataFrame(
        {
            "age": ["63"],
            "trestbps": ["145"],
            "chol": ["233"],
        }
    )

    cleaned = clean_heart_disease_data(data)

    assert cleaned["age"].dtype.kind in "fi"
    assert cleaned["trestbps"].dtype.kind in "fi"
    assert cleaned["chol"].dtype.kind in "fi"