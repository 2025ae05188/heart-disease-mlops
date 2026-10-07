"""Evaluation plots and report generation."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
)


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """
    Generate and save a confusion matrix plot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_roc_auc_curve(
    y_true: Any,
    y_prob: Any,
    output_path: str | Path,
    title: str = "ROC-AUC Curve",
) -> Path:
    """
    Generate and save a ROC-AUC curve.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
    )

    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path

def save_classification_report(
    y_true: Any,
    y_pred: Any,
    output_path: str | Path,
) -> Path:
    """
    Generate and save the sklearn classification report as a text file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0,
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )

    return output_path

def generate_evaluation_artifacts(
    model: Any,
    X: Any,
    y: Any,
    output_dir: str | Path,
    model_name: str = "model",
) -> dict[str, Path]:
    """
    Generate evaluation plots and reports for a trained model.

    Returns
    -------
    dict[str, Path]
        Paths to generated evaluation artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    confusion_matrix_path = plot_confusion_matrix(
        y_true=y,
        y_pred=y_pred,
        output_path=output_dir / f"{model_name}_confusion_matrix.png",
        title=f"{model_name} - Confusion Matrix",
    )

    classification_report_path = save_classification_report(
        y_true=y,
        y_pred=y_pred,
        output_path=output_dir / f"{model_name}_classification_report.txt",
    )

    roc_curve_path = plot_roc_auc_curve(
        y_true=y,
        y_prob=y_prob,
        output_path=output_dir / f"{model_name}_roc_auc_curve.png",
        title=f"{model_name} - ROC-AUC Curve",
    )

    return {
        "confusion_matrix": confusion_matrix_path,
        "classification_report": classification_report_path,
        "roc_auc_curve": roc_curve_path,
    }