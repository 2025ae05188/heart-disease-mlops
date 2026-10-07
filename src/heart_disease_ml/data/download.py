"""Utilities for downloading the UCI Heart Disease dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


UCI_HEART_DISEASE_ID = 45


def download_heart_disease(output_path: str | Path | None = None) -> pd.DataFrame:
    """
    Fetch the UCI Heart Disease dataset.

    Parameters
    ----------
    output_path:
        Optional path where the downloaded feature and target data
        should be saved as a CSV file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing features and the target column.
    """
    heart_disease = fetch_ucirepo(id=UCI_HEART_DISEASE_ID)

    features = heart_disease.data.features.copy()
    target = heart_disease.data.targets.copy()

    data = pd.concat([features, target], axis=1)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)

    return data