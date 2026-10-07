"""Tests for data loading and validation."""

from pathlib import Path

import pandas as pd

from heart_disease_ml.preprocessing.cleaning import (
    clean_heart_disease_data,
)
from heart_disease_ml.preprocessing.features import (
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"


def test_dataset_exists():
    """The raw dataset should exist."""
    assert DATA_PATH.exists()


def test_dataset_is_not_empty():
    """The raw dataset should contain observations."""
    data = pd.read_csv(DATA_PATH)

    assert not data.empty
    assert len(data) > 0


def test_target_is_binary():
    """The prepared target should contain only 0 and 1."""
    data = pd.read_csv(DATA_PATH)
    data = clean_heart_disease_data(data)

    _, y = split_features_target(data)

    assert set(y.unique()).issubset({0, 1})
    assert len(y) == len(data)