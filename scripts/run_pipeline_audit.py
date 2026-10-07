"""Run an end-to-end audit of the existing ML pipeline."""

from sklearn.model_selection import train_test_split

from heart_disease_ml.data.load import load_csv
from heart_disease_ml.preprocessing.cleaning import clean_heart_disease_data
from heart_disease_ml.preprocessing.features import split_features_target
from heart_disease_ml.models.factory import create_model_pipeline
from heart_disease_ml.models.train import train_model
from heart_disease_ml.evaluation.cross_validation import cross_validate_model
from heart_disease_ml.evaluation.metrics import evaluate_model
from heart_disease_ml.evaluation.reports import generate_evaluation_artifacts
from heart_disease_ml.utils.paths import RAW_DATA_DIR, ARTIFACTS_DIR


RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    """Run the complete ML pipeline without MLflow."""

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------
    data_path = RAW_DATA_DIR / "heart_disease.csv"

    print("\n[1/7] Loading dataset...")
    data = load_csv(data_path)
    print(f"Dataset shape: {data.shape}")

    # ---------------------------------------------------------
    # 2. Clean data
    # ---------------------------------------------------------
    print("\n[2/7] Cleaning dataset...")
    data = clean_heart_disease_data(data)
    print(f"Cleaned shape: {data.shape}")

    # ---------------------------------------------------------
    # 3. Prepare features and target
    # ---------------------------------------------------------
    print("\n[3/7] Preparing features and target...")
    X, y = split_features_target(data)

    print(f"Features shape: {X.shape}")
    print(f"Target shape:   {y.shape}")
    print(f"Class distribution:\n{y.value_counts().sort_index()}")

    # ---------------------------------------------------------
    # 4. Train/test split
    # ---------------------------------------------------------
    print("\n[4/7] Splitting train/test data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")

    # ---------------------------------------------------------
    # 5. Train and evaluate both models
    # ---------------------------------------------------------
    for model_name in ["logistic_regression", "random_forest"]:

        print("\n" + "=" * 70)
        print(f"MODEL: {model_name}")
        print("=" * 70)

        # Create pipeline
        model = create_model_pipeline(model_name)

        # Train
        print("\n[5/7] Training...")
        trained_model = train_model(
            model,
            X_train,
            y_train,
        )

        print("Training completed.")

        # -----------------------------------------------------
        # Cross-validation
        # -----------------------------------------------------
        print("\n[6/7] Running 5-fold cross-validation...")

        cv_results = cross_validate_model(
            trained_model,
            X_train,
            y_train,
        )

        print("\nCross-validation results:")

        for metric, values in cv_results.items():
            if metric.endswith("_mean"):
                std_key = metric.replace("_mean", "_std")

                print(
                    f"  {metric.replace('_mean', ''):10s}: "
                    f"{values:.4f} ± {cv_results[std_key]:.4f}"
                )

        # -----------------------------------------------------
        # Test evaluation
        # -----------------------------------------------------
        print("\n[7/7] Evaluating on test set...")

        test_metrics = evaluate_model(
            trained_model,
            X_test,
            y_test,
        )

        print("\nTest metrics:")

        for metric, value in test_metrics.items():
            print(f"  {metric:10s}: {value:.4f}")

        # -----------------------------------------------------
        # Generate evaluation artifacts
        # -----------------------------------------------------
        model_artifact_dir = ARTIFACTS_DIR / "pipeline_audit" / model_name
        model_artifact_dir.mkdir(parents=True, exist_ok=True)

        generate_evaluation_artifacts(
            trained_model,
            X_test,
            y_test,
            output_dir=model_artifact_dir,
            model_name=model_name,
        )

        print("\nArtifacts saved to:")
        print(f"  {model_artifact_dir}")


if __name__ == "__main__":
    main()