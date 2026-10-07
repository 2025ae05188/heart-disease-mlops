"""Model definitions and default configurations."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


RANDOM_STATE = 42


def create_logistic_regression(**kwargs) -> LogisticRegression:
    """
    Create the baseline Logistic Regression model.
    """
    params = {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    }
    params.update(kwargs)

    return LogisticRegression(**params)


def create_random_forest(**kwargs) -> RandomForestClassifier:
    """
    Create the baseline Random Forest model.
    """
    params = {
        "random_state": RANDOM_STATE,
    }
    params.update(kwargs)

    return RandomForestClassifier(**params)