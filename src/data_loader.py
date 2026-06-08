"""Data loading utilities for TON_IoT, CICIoT, and synthetic datasets."""

from pathlib import Path

import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file from a project-relative or absolute path."""
    return pd.read_csv(Path(path))
