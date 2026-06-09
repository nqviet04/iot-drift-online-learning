"""Recent-buffer management and adaptive Random Forest retraining."""

import json
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.config import CLOUD_MODEL_DIR, RANDOM_STATE, RECENT_BUFFER_SIZE
from src.static_model import save_model, train_random_forest


RETRAIN_LOG_COLUMNS = [
    "detected_drift_index",
    "version",
    "retrain_time_seconds",
    "n_samples",
    "model_path",
]


class RecentBuffer:
    """FIFO buffer containing the most recent feature rows and labels."""

    def __init__(self, max_size: int = RECENT_BUFFER_SIZE) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be greater than 0.")

        self.max_size = max_size
        self._X: deque[np.ndarray] = deque(maxlen=max_size)
        self._y: deque[Any] = deque(maxlen=max_size)
        self._n_features: int | None = None

    def add(self, x, y) -> None:
        """Add one sample and automatically discard the oldest when full."""
        if isinstance(x, pd.Series):
            row = x.to_numpy(dtype=float)
        elif isinstance(x, pd.DataFrame):
            if len(x) != 1:
                raise ValueError("RecentBuffer.add expects exactly one DataFrame row.")
            row = x.iloc[0].to_numpy(dtype=float)
        else:
            row = np.asarray(x, dtype=float).reshape(-1)

        if row.size == 0:
            raise ValueError("Cannot add an empty feature row to RecentBuffer.")
        if self._n_features is None:
            self._n_features = row.size
        elif row.size != self._n_features:
            raise ValueError(
                f"Feature size mismatch: expected {self._n_features}, got {row.size}."
            )

        self._X.append(row.copy())
        self._y.append(y)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return buffered samples in chronological order."""
        if not self._X:
            n_features = self._n_features or 0
            return np.empty((0, n_features), dtype=float), np.empty(0)

        return np.vstack(self._X), np.asarray(self._y)

    def __len__(self) -> int:
        """Return the current number of buffered samples."""
        return len(self._y)


class AdaptiveRandomForestTrainer:
    """Own the active Random Forest model and create versioned retrains."""

    def __init__(
        self,
        model,
        model_version: int = 0,
        random_state: int = RANDOM_STATE,
        model_storage_dir: str | Path = CLOUD_MODEL_DIR,
    ) -> None:
        if model_version < 0:
            raise ValueError("model_version cannot be negative.")

        self.model = model
        self.model_version = model_version
        self.random_state = random_state
        self.model_storage_dir = Path(model_storage_dir)

    @property
    def version(self) -> int:
        """Return the current model version."""
        return self.model_version

    def retrain(self, X_recent, y_recent) -> dict[str, Any]:
        """Train a new model on recent data, version it, and save it locally."""
        if len(X_recent) == 0:
            raise ValueError("Cannot retrain with an empty recent buffer.")
        if len(X_recent) != len(y_recent):
            raise ValueError("X_recent and y_recent must have the same number of rows.")

        start_time = perf_counter()
        new_model = train_random_forest(
            X_recent,
            y_recent,
            random_state=self.random_state,
        )
        retrain_time = perf_counter() - start_time

        new_version = self.model_version + 1
        model_path = self.model_storage_dir / f"adaptive_rf_v{new_version}.joblib"
        save_model(new_model, model_path)

        self.model = new_model
        self.model_version = new_version

        return {
            "version": new_version,
            "retrain_time_seconds": retrain_time,
            "n_samples": len(y_recent),
            "model_path": str(model_path),
        }


def save_retraining_log(
    records: list[dict[str, Any]],
    csv_path: str | Path,
    json_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Save retraining records to CSV and optionally JSON."""
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    if records:
        log_df = pd.DataFrame(records)
        ordered_columns = [
            column for column in RETRAIN_LOG_COLUMNS if column in log_df.columns
        ]
        remaining_columns = [
            column for column in log_df.columns if column not in ordered_columns
        ]
        log_df = log_df[ordered_columns + remaining_columns]
    else:
        log_df = pd.DataFrame(columns=RETRAIN_LOG_COLUMNS)

    log_df.to_csv(csv_output, index=False)

    json_output: Path | None = None
    if json_path is not None:
        json_output = Path(json_path)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return csv_output, json_output
