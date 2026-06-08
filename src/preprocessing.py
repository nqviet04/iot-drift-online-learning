"""Preprocessing utilities for IoT binary classification datasets."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import BINARY_LABEL_COLUMN


COMMON_LABEL_COLUMNS = [
    BINARY_LABEL_COLUMN,
    "label",
    "Label",
    "type",
    "attack_type",
    "category",
    "class",
]
BENIGN_LABEL_VALUES = {
    "0",
    "0.0",
    "benign",
    "benigntraffic",
    "benign traffic",
    "normal",
    "normaltraffic",
    "normal traffic",
    "false",
}
NON_FEATURE_EXACT_NAMES = {
    "timestamp",
    "date",
    "time",
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "ip",
}


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect a likely label column, preferring label_binary when available.

    Raises:
        ValueError: If no common label column can be found.
    """
    if df.empty:
        raise ValueError("Cannot detect label column from an empty DataFrame.")

    if BINARY_LABEL_COLUMN in df.columns:
        return BINARY_LABEL_COLUMN

    for candidate in COMMON_LABEL_COLUMNS:
        if candidate in df.columns:
            return candidate

    lower_to_original = {column.lower(): column for column in df.columns}
    for candidate in COMMON_LABEL_COLUMNS:
        match = lower_to_original.get(candidate.lower())
        if match is not None:
            return match

    raise ValueError(
        "Could not detect a label column. Expected one of: "
        f"{COMMON_LABEL_COLUMNS}. Available columns: {list(df.columns)}"
    )


def _map_to_binary_label(value: object) -> int:
    """Map benign/normal/0 labels to 0 and every other value to 1."""
    if pd.isna(value):
        return 1

    if isinstance(value, (bool, np.bool_)):
        return int(value)

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return 0 if float(value) == 0.0 else 1

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")
    compact = normalized.replace(" ", "")
    return 0 if normalized in BENIGN_LABEL_VALUES or compact in BENIGN_LABEL_VALUES else 1


def create_binary_label(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Create or overwrite label_binary using a common benign-vs-attack mapping.

    Values equal to benign, normal, or 0 are mapped to 0. All remaining values
    are mapped to 1. Existing columns, including attack_type, are preserved.
    """
    if df.empty:
        raise ValueError("Cannot create labels from an empty DataFrame.")

    detected_label_col = label_col or detect_label_column(df)
    if detected_label_col not in df.columns:
        raise ValueError(
            f"Label column '{detected_label_col}' was not found. "
            f"Available columns: {list(df.columns)}"
        )

    output = df.copy()
    output[BINARY_LABEL_COLUMN] = output[detected_label_col].map(_map_to_binary_label).astype(int)
    return output


def _is_non_feature_column(column: str, series: pd.Series, label_col: str) -> bool:
    """Return True for metadata, label, or hard-to-handle object columns."""
    column_lower = column.strip().lower()
    label_names = {name.lower() for name in COMMON_LABEL_COLUMNS}

    if column == label_col or column_lower in label_names:
        return True
    if column_lower in NON_FEATURE_EXACT_NAMES:
        return True
    if "timestamp" in column_lower:
        return True
    if "ip" in column_lower and not pd.api.types.is_numeric_dtype(series):
        return True
    if "port" in column_lower and not pd.api.types.is_numeric_dtype(series):
        return True

    return False


def clean_features(
    df: pd.DataFrame,
    label_col: str = BINARY_LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build numeric model features and binary target from a dataset.

    The function does not fit any scaler. Fit scaling only on the train split
    with Preprocessor.fit_transform, then call Preprocessor.transform for stream
    or test windows to avoid data leakage.
    """
    if df.empty:
        raise ValueError("Cannot clean features from an empty DataFrame.")
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' was not found. "
            "Call create_binary_label first or pass the correct label_col."
        )

    y = df[label_col].map(_map_to_binary_label).astype(int)
    candidate_columns = [
        column
        for column in df.columns
        if not _is_non_feature_column(column, df[column], label_col)
    ]
    numeric_columns = [
        column
        for column in candidate_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if not numeric_columns:
        raise ValueError(
            "No numeric feature columns remain after preprocessing. "
            "Check dataset columns or add numeric feature engineering."
        )

    X = df[numeric_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    X = X.astype(float)

    return X, y, numeric_columns


class Preprocessor:
    """StandardScaler wrapper for train-only fitting and reusable transforms."""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_names: list[str] | None = None
        self.is_fitted = False

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """Fit the scaler on train data only and return scaled train features."""
        self._validate_features(X_train)
        self.feature_names = list(X_train.columns)
        transformed = self.scaler.fit_transform(X_train)
        self.is_fitted = True
        return pd.DataFrame(transformed, columns=self.feature_names, index=X_train.index)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the scaler fitted on the train split."""
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Preprocessor is not fitted. Call fit_transform on X_train first.")

        missing_columns = [column for column in self.feature_names if column not in X.columns]
        if missing_columns:
            raise ValueError(f"Input data is missing required feature columns: {missing_columns}")

        ordered_X = X[self.feature_names]
        transformed = self.scaler.transform(ordered_X)
        return pd.DataFrame(transformed, columns=self.feature_names, index=X.index)

    def save(self, path: str | Path) -> Path:
        """Save this fitted preprocessor to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted Preprocessor.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "Preprocessor":
        """Load a previously saved preprocessor."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {input_path}")

        preprocessor = joblib.load(input_path)
        if not isinstance(preprocessor, cls):
            raise TypeError(f"File does not contain a {cls.__name__}: {input_path}")
        return preprocessor

    @staticmethod
    def _validate_features(X: pd.DataFrame) -> None:
        """Validate feature matrix before scaling."""
        if X.empty:
            raise ValueError("Feature matrix is empty.")
        if not all(pd.api.types.is_numeric_dtype(X[column]) for column in X.columns):
            raise ValueError("All features must be numeric before scaling.")


def normalize_binary_labels(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Backward-compatible alias for early skeleton code."""
    return create_binary_label(df, label_col=label_column)
