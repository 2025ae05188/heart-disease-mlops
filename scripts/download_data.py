"""Download the UCI Heart Disease dataset."""

from heart_disease_ml.data.download import download_heart_disease
from heart_disease_ml.utils.paths import RAW_DATA_DIR


OUTPUT_PATH = RAW_DATA_DIR / "heart_disease.csv"


def main() -> None:
    """Download and save the raw dataset."""

    data = download_heart_disease(OUTPUT_PATH)

    print(f"Saved dataset: {OUTPUT_PATH}")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")


if __name__ == "__main__":
    main()