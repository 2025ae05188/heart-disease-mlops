"""Set up and validate the project environment."""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> None:
    """Run a command and stop if it fails."""
    print(f"\nRunning: {' '.join(command)}")
    subprocess.run(command, check=True)


def verify_python_version() -> None:
    """Verify that the Python version is supported."""
    major = sys.version_info.major
    minor = sys.version_info.minor

    if (major, minor) != (3, 12):
        raise RuntimeError(
            f"Python 3.12 is required. Found Python {major}.{minor}."
        )

    print(f"Python version OK: {major}.{minor}")


def install_requirements(requirements_file: str) -> None:
    """Install dependencies from a requirements file."""
    requirements_path = PROJECT_ROOT / requirements_file

    if not requirements_path.exists():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_path}"
        )

    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    )


def create_project_directories() -> None:
    """Create directories required by the project."""
    directories = [
        "data/raw",
        "data/processed",
        "artifacts",
        "models",
    ]

    for directory in directories:
        path = PROJECT_ROOT / directory
        path.mkdir(parents=True, exist_ok=True)

    print("Project directories verified.")


def main() -> None:
    """Set up the base project environment."""
    print("========================================")
    print("Heart Disease MLOps - Environment Setup")
    print("========================================")

    verify_python_version()

    print("\nInstalling base requirements...")
    install_requirements("requirements.txt")

    create_project_directories()

    print("\nSetup completed successfully.")


if __name__ == "__main__":
    main()