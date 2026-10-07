"""Tests for model construction and prediction."""

from pathlib import Path

from heart_disease_ml.data.load import load_csv
from heart_disease_ml.models.factory import create_model_pipeline
from heart_disease_ml.preprocessing.cleaning import (
    clean_heart_disease_data,
)
from heart_disease_ml.preprocessing.features import (
    split_features_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"


def test_logistic_regression_pipeline_can_train():
    """The Logistic Regression pipeline should train successfully."""

    data = load_csv(DATA_PATH)
    data = clean_heart_disease_data(data)
    X, y = split_features_target(data)

    model = create_model_pipeline("logistic_regression")

    model.fit(X, y)

    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    assert set(predictions).issubset({0, 1})


def test_logistic_regression_pipeline_can_predict():
    """The Logistic Regression pipeline should generate predictions."""

    data = load_csv(DATA_PATH)
    data = clean_heart_disease_data(data)
    X, y = split_features_target(data)

    model = create_model_pipeline("logistic_regression")
    model.fit(X, y)

    sample = X.head(1)

    prediction = model.predict(sample)
    probabilities = model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in {0, 1}
    assert probabilities.shape == (1, 2)
    assert 0.0 <= probabilities[0].max() <= 1.0