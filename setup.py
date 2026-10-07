from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).resolve().parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"


def read_requirements():
    """Read runtime dependencies from requirements.txt."""
    with REQUIREMENTS_FILE.open("r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file
            if line.strip() and not line.startswith("#")
        ]


setup(
    name="heart-disease-mlops",
    version="0.1.0",
    description="End-to-end MLOps project for UCI Heart Disease classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    python_requires=">=3.12,<3.13",
)