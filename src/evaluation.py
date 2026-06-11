"""Evaluation helpers for static and adaptive stream experiments."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(y_true, y_pred) -> dict[str, Any]:
    """Compute binary classification metrics with safe zero-division behavior."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm.tolist(),
    }


def _slice_rows(data, start: int, end: int):
    """Slice pandas or array-like data by row position."""
    if hasattr(data, "iloc"):
        return data.iloc[start:end]
    return data[start:end]


def evaluate_by_windows(
    model,
    X_stream,
    y_stream,
    window_size: int = 1000,
) -> pd.DataFrame:
    """Evaluate a model on consecutive stream windows without shuffling."""
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0.")

    n_rows = len(X_stream)
    if n_rows == 0:
        raise ValueError("X_stream is empty.")
    if len(y_stream) != n_rows:
        raise ValueError("X_stream and y_stream must have the same number of rows.")

    rows: list[dict[str, float | int]] = []
    for window_id, start_index in enumerate(range(0, n_rows, window_size)):
        end_exclusive = min(start_index + window_size, n_rows)
        X_window = _slice_rows(X_stream, start_index, end_exclusive)
        y_window = _slice_rows(y_stream, start_index, end_exclusive)
        y_pred = model.predict(X_window)
        metrics = compute_classification_metrics(y_window, y_pred)

        rows.append(
            {
                "window_id": window_id,
                "start_index": start_index,
                "end_index": end_exclusive - 1,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
            }
        )

    return pd.DataFrame(rows)


def _json_default(value: Any) -> Any:
    """Convert numpy/pandas objects to JSON-serializable Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_metrics_json(metrics: dict[str, Any], path: str | Path) -> Path:
    """Save metrics as a JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def save_window_metrics_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save per-window metrics to CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def _point_to_window_position(window_metrics_df: pd.DataFrame, point: int) -> float | None:
    """Map a stream row index to a fractional window_id position for plotting."""
    if window_metrics_df.empty:
        return None

    for row in window_metrics_df.itertuples(index=False):
        if row.start_index <= point <= row.end_index:
            width = max(row.end_index - row.start_index + 1, 1)
            fraction = (point - row.start_index) / width
            return row.window_id + fraction
    return None


def _draw_drift_lines(
    ax,
    window_metrics_df: pd.DataFrame,
    points: list[int] | None,
    color: str,
    linestyle: str,
    label: str,
) -> None:
    """Draw drift markers on a metric-over-time plot."""
    if not points:
        return

    label_used = False
    for point in points:
        position = _point_to_window_position(window_metrics_df, int(point))
        if position is None:
            continue
        ax.axvline(
            x=position,
            color=color,
            linestyle=linestyle,
            linewidth=1.5,
            alpha=0.8,
            label=label if not label_used else None,
        )
        label_used = True


def plot_metric_over_time(
    window_metrics_df: pd.DataFrame,
    metric_name: str,
    output_path: str | Path,
    drift_points: list[int] | None = None,
    detected_drifts: list[int] | None = None,
) -> Path:
    """Plot a window-level metric and optional actual/detected drift markers."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to plot metrics. Install project dependencies "
            "with: pip install -r requirements.txt"
        ) from exc

    required_columns = {"window_id", "start_index", "end_index", metric_name}
    missing_columns = required_columns - set(window_metrics_df.columns)
    if missing_columns:
        raise ValueError(f"window_metrics_df is missing columns: {sorted(missing_columns)}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        window_metrics_df["window_id"],
        window_metrics_df[metric_name],
        marker="o",
        linewidth=2,
        label=metric_name,
    )
    _draw_drift_lines(ax, window_metrics_df, drift_points, "tab:red", "--", "Actual drift")
    _draw_drift_lines(
        ax,
        window_metrics_df,
        detected_drifts,
        "tab:purple",
        ":",
        "Detected drift",
    )

    ax.set_title(f"{metric_name.upper()} over stream windows")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(metric_name)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return output


def classification_metrics(y_true, y_pred) -> dict[str, Any]:
    """Compute binary classification metrics."""
    return compute_classification_metrics(y_true, y_pred)
