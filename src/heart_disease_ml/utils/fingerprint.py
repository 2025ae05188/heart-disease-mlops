"""Utilities for generating dataset fingerprints."""

import hashlib
from pathlib import Path


def fingerprint_file(
    path: str | Path,
    algorithm: str = "md5",
) -> str:
    """Return a hash fingerprint for a file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = hashlib.new(algorithm)

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()