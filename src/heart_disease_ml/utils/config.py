"""Project configuration utilities."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_project_config() -> dict[str, Any]:
    """Load project-level configuration."""

    return load_yaml(CONFIG_DIR / "project.yaml")


def load_data_config() -> dict[str, Any]:
    """Load dataset configuration."""

    return load_yaml(CONFIG_DIR / "data.yaml")


def load_model_config() -> dict[str, Any]:
    """Load model and training configuration."""

    return load_yaml(CONFIG_DIR / "models.yaml")


def load_mlflow_config() -> dict[str, Any]:
    """Load MLflow configuration."""

    return load_yaml(CONFIG_DIR / "mlflow.yaml")