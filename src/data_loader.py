"""Dataset loading utilities for IoT drift detection experiments."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DATA_SYNTHETIC_DIR, TON_IOT_RAW_DIR, CIC_IOT_RAW_DIR, TRAIN_RATIO


SYNTHETIC_DATASET_FILENAME = "synthetic_iot_drift.csv"


def _safe_print(message: Any) -> None:
    """Print text safely on Windows consoles with limited encodings."""
    text = str(message)
    encoding = getattr(__import__("sys").stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _resolve_existing_csv(path: str | Path) -> Path:
    """Resolve a CSV path and raise a clear error when it is missing."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}. Please check the filename and dataset location."
        )
    if not csv_path.is_file():
        raise ValueError(f"Expected a CSV file path, but got a directory: {csv_path}")
    return csv_path


def _print_basic_info(df: pd.DataFrame) -> None:
    """Print a compact dataset summary useful for quick pipeline checks."""
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    _safe_print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    _safe_print(f"Columns: {list(df.columns)}")

    if missing_values.empty:
        _safe_print("Missing values: none")
    else:
        _safe_print("Missing values by column:")
        _safe_print(missing_values.to_string())


def load_csv_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset, validate it exists, and print basic information.

    Args:
        path: CSV file path. It can be absolute or relative to the current
            working directory.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the path points to a directory or pandas cannot parse it.
    """
    csv_path = _resolve_existing_csv(path)

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV file: {csv_path}") from exc

    _print_basic_info(df)
    return df


def load_synthetic_dataset() -> pd.DataFrame:
    """Load the synthetic IoT drift dataset, generating it when missing."""
    csv_path = DATA_SYNTHETIC_DIR / SYNTHETIC_DATASET_FILENAME

    if not csv_path.exists():
        _safe_print("Synthetic dataset not found. Generating a new synthetic IoT drift dataset...")
        from src.drift_simulator import save_synthetic_dataset

        save_synthetic_dataset(csv_path=csv_path)

    return load_csv_dataset(str(csv_path))


def load_ton_iot_dataset(filename: str) -> pd.DataFrame:
    """Load a TON_IoT CSV file from data/raw/TON_IoT without schema assumptions."""
    csv_path = TON_IOT_RAW_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"TON_IoT file not found: {csv_path}. "
            "Place the file under data/raw/TON_IoT/ and pass its filename."
        )
    return load_csv_dataset(str(csv_path))


def load_cic_iot_dataset(filename: str) -> pd.DataFrame:
    """Load a CICIoT CSV file from data/raw/CICIoT without schema assumptions."""
    csv_path = CIC_IOT_RAW_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CICIoT file not found: {csv_path}. "
            "Place the file under data/raw/CICIoT/ and pass its filename."
        )
    return load_csv_dataset(str(csv_path))


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    timestamp_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into initial train data and stream data without shuffling.

    If timestamp_col is provided, rows are sorted by that column first. If it is
    omitted, the original row order is treated as stream order.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise ValueError(
                f"timestamp_col '{timestamp_col}' was not found in columns: {list(df.columns)}"
            )
        ordered_df = df.sort_values(timestamp_col).reset_index(drop=True)
    else:
        ordered_df = df.reset_index(drop=True)

    split_index = int(len(ordered_df) * train_ratio)
    if split_index == 0 or split_index == len(ordered_df):
        raise ValueError(
            "Split would produce an empty train or stream set. "
            "Use more rows or adjust train_ratio."
        )

    train_df = ordered_df.iloc[:split_index].copy()
    stream_df = ordered_df.iloc[split_index:].copy()
    return train_df, stream_df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Backward-compatible alias for early skeleton code."""
    return load_csv_dataset(str(path))
