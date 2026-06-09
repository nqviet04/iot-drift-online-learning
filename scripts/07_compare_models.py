"""Compare static and adaptive model experiment results."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FIGURE_DIR, METRIC_DIR


STATIC_WINDOW_PATH = METRIC_DIR / "static_window_metrics.csv"
ADAPTIVE_STATIC_WINDOW_PATH = METRIC_DIR / "adaptive_static_window_metrics.csv"
ADAPTIVE_LSTM_WINDOW_PATH = METRIC_DIR / "adaptive_lstm_window_metrics.csv"
ADAPTIVE_STATIC_RETRAIN_PATH = METRIC_DIR / "adaptive_static_retrain_log.csv"
ADAPTIVE_LSTM_RETRAIN_PATH = METRIC_DIR / "adaptive_lstm_retrain_log.csv"

SUMMARY_CSV_PATH = METRIC_DIR / "model_comparison_summary.csv"
SUMMARY_JSON_PATH = METRIC_DIR / "model_comparison_summary.json"

F1_OVER_TIME_PATH = FIGURE_DIR / "compare_f1_over_time.png"
ACCURACY_OVER_TIME_PATH = FIGURE_DIR / "compare_accuracy_over_time.png"
AVERAGE_F1_PATH = FIGURE_DIR / "compare_average_f1.png"
UPDATE_COST_PATH = FIGURE_DIR / "accuracy_vs_update_cost.png"

SUMMARY_COLUMNS = [
    "model_name",
    "average_accuracy",
    "average_precision",
    "average_recall",
    "average_f1",
    "final_f1",
    "worst_f1",
    "retrain_count",
    "total_retrain_time_seconds",
    "average_retrain_time_seconds",
]


def _warn(message: str) -> None:
    """Emit a visible warning without stopping the comparison."""
    warnings.warn(message, stacklevel=2)


def _read_csv_if_available(path: Path, description: str) -> pd.DataFrame | None:
    """Read a CSV when available and return None after a friendly warning."""
    if not path.exists():
        _warn(f"Missing {description}: {path}")
        return None

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        _warn(f"Could not read {description} at {path}: {exc}")
        return None


def _extract_model_metrics(
    df: pd.DataFrame | None,
    model_name: str,
    prefix: str = "",
) -> pd.DataFrame | None:
    """Normalize one model's window metrics to common column names."""
    if df is None:
        return None

    source_columns = {
        "accuracy": f"{prefix}accuracy",
        "precision": f"{prefix}precision",
        "recall": f"{prefix}recall",
        "f1": f"{prefix}f1",
    }
    required_columns = {"window_id", *source_columns.values()}
    missing = required_columns - set(df.columns)
    if missing:
        _warn(
            f"Skipping {model_name}: window metrics are missing columns "
            f"{sorted(missing)}."
        )
        return None

    normalized = pd.DataFrame(
        {
            "window_id": df["window_id"],
            "accuracy": df[source_columns["accuracy"]],
            "precision": df[source_columns["precision"]],
            "recall": df[source_columns["recall"]],
            "f1": df[source_columns["f1"]],
        }
    )
    normalized["model_name"] = model_name
    return normalized


def _retrain_statistics(
    retrain_df: pd.DataFrame | None,
    model_name: str,
    static_model: bool = False,
) -> dict[str, float | int]:
    """Calculate retraining count and timing statistics."""
    if static_model:
        return {
            "retrain_count": 0,
            "total_retrain_time_seconds": 0.0,
            "average_retrain_time_seconds": 0.0,
        }

    if retrain_df is None:
        _warn(f"Retraining statistics are unavailable for {model_name}.")
        return {
            "retrain_count": np.nan,
            "total_retrain_time_seconds": np.nan,
            "average_retrain_time_seconds": np.nan,
        }

    if "retrain_time_seconds" not in retrain_df.columns:
        _warn(
            f"Retrain log for {model_name} has no 'retrain_time_seconds' column."
        )
        return {
            "retrain_count": len(retrain_df),
            "total_retrain_time_seconds": np.nan,
            "average_retrain_time_seconds": np.nan,
        }

    retrain_times = pd.to_numeric(
        retrain_df["retrain_time_seconds"],
        errors="coerce",
    ).dropna()
    return {
        "retrain_count": int(len(retrain_df)),
        "total_retrain_time_seconds": float(retrain_times.sum()),
        "average_retrain_time_seconds": (
            float(retrain_times.mean()) if not retrain_times.empty else 0.0
        ),
    }


def _build_summary_row(
    metrics_df: pd.DataFrame,
    model_name: str,
    retrain_stats: dict[str, float | int],
) -> dict[str, float | int | str]:
    """Build one model-comparison summary row."""
    return {
        "model_name": model_name,
        "average_accuracy": float(metrics_df["accuracy"].mean()),
        "average_precision": float(metrics_df["precision"].mean()),
        "average_recall": float(metrics_df["recall"].mean()),
        "average_f1": float(metrics_df["f1"].mean()),
        "final_f1": float(metrics_df["f1"].iloc[-1]),
        "worst_f1": float(metrics_df["f1"].min()),
        **retrain_stats,
    }


def _get_pyplot():
    """Load matplotlib with a non-interactive backend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for comparison plots. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return plt


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    """Create a valid placeholder plot when no usable data is available."""
    plt = _get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metric_over_time(
    model_metrics: list[pd.DataFrame],
    metric: str,
    path: Path,
) -> None:
    """Plot one metric over windows for all available models."""
    if not model_metrics:
        _save_empty_plot(path, f"{metric.title()} over time", "No metrics available")
        return

    plt = _get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for metrics_df in model_metrics:
        model_name = str(metrics_df["model_name"].iloc[0])
        ax.plot(
            metrics_df["window_id"],
            metrics_df[metric],
            marker="o",
            linewidth=2,
            label=model_name,
        )

    ax.set_title(f"{metric.title()} over stream windows")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(metric.title())
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_average_f1(summary_df: pd.DataFrame) -> None:
    """Plot average F1 as a bar chart."""
    if summary_df.empty:
        _save_empty_plot(AVERAGE_F1_PATH, "Average F1", "No summary data available")
        return

    plt = _get_pyplot()
    AVERAGE_F1_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        summary_df["model_name"],
        summary_df["average_f1"],
        color=["tab:blue", "tab:green", "tab:orange"][: len(summary_df)],
    )
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title("Average F1 comparison")
    ax.set_ylabel("Average F1")
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(AVERAGE_F1_PATH, dpi=150)
    plt.close(fig)


def _plot_update_cost_tradeoff(summary_df: pd.DataFrame) -> None:
    """Plot average F1 against total model-update time."""
    plot_df = summary_df.dropna(
        subset=["average_f1", "total_retrain_time_seconds"]
    )
    if plot_df.empty:
        _save_empty_plot(
            UPDATE_COST_PATH,
            "Average F1 vs update cost",
            "No update-cost data available",
        )
        return

    plt = _get_pyplot()
    UPDATE_COST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = [
        {"marker": "o", "color": "tab:blue", "facecolors": "none", "s": 150},
        {"marker": "x", "color": "tab:green", "s": 120},
        {"marker": "D", "color": "tab:orange", "s": 90},
    ]
    for row, style in zip(plot_df.itertuples(index=False), styles):
        ax.scatter(
            row.total_retrain_time_seconds,
            row.average_f1,
            label=row.model_name,
            linewidths=2,
            **style,
        )

    ax.set_title("Average F1 vs total retraining time")
    ax.set_xlabel("Total retraining time (seconds)")
    ax.set_ylabel("Average F1")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(UPDATE_COST_PATH, dpi=150)
    plt.close(fig)


def _json_records(summary_df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to strict JSON records, replacing NaN with null."""
    clean_df = summary_df.astype(object).where(pd.notna(summary_df), None)
    return clean_df.to_dict(orient="records")


def main() -> None:
    """Load available experiment outputs and create comparison artifacts."""
    static_window = _read_csv_if_available(
        STATIC_WINDOW_PATH,
        "static model window metrics",
    )
    adaptive_static_window = _read_csv_if_available(
        ADAPTIVE_STATIC_WINDOW_PATH,
        "adaptive static window metrics",
    )
    adaptive_lstm_window = _read_csv_if_available(
        ADAPTIVE_LSTM_WINDOW_PATH,
        "adaptive LSTM window metrics",
    )
    adaptive_static_retrain = _read_csv_if_available(
        ADAPTIVE_STATIC_RETRAIN_PATH,
        "adaptive static retrain log",
    )
    adaptive_lstm_retrain = _read_csv_if_available(
        ADAPTIVE_LSTM_RETRAIN_PATH,
        "adaptive LSTM retrain log",
    )

    model_specs = [
        (
            "Static Random Forest",
            _extract_model_metrics(
                static_window,
                "Static Random Forest",
            ),
            _retrain_statistics(None, "Static Random Forest", static_model=True),
        ),
        (
            "Adaptive Random Forest",
            _extract_model_metrics(
                adaptive_static_window,
                "Adaptive Random Forest",
                prefix="adaptive_",
            ),
            _retrain_statistics(
                adaptive_static_retrain,
                "Adaptive Random Forest",
            ),
        ),
        (
            "Adaptive LSTM",
            _extract_model_metrics(
                adaptive_lstm_window,
                "Adaptive LSTM",
                prefix="adaptive_",
            ),
            _retrain_statistics(
                adaptive_lstm_retrain,
                "Adaptive LSTM",
            ),
        ),
    ]

    normalized_metrics: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    for model_name, metrics_df, retrain_stats in model_specs:
        if metrics_df is None or metrics_df.empty:
            _warn(f"No usable window metrics for {model_name}; skipping summary row.")
            continue
        normalized_metrics.append(metrics_df)
        summary_rows.append(
            _build_summary_row(metrics_df, model_name, retrain_stats)
        )

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    SUMMARY_JSON_PATH.write_text(
        json.dumps(_json_records(summary_df), indent=2),
        encoding="utf-8",
    )

    _plot_metric_over_time(normalized_metrics, "f1", F1_OVER_TIME_PATH)
    _plot_metric_over_time(
        normalized_metrics,
        "accuracy",
        ACCURACY_OVER_TIME_PATH,
    )
    _plot_average_f1(summary_df)
    _plot_update_cost_tradeoff(summary_df)

    print("Model comparison completed.")
    print(f"Models compared: {len(summary_df)}")
    print(f"Summary CSV: {SUMMARY_CSV_PATH.relative_to(ROOT_DIR)}")
    print(f"Summary JSON: {SUMMARY_JSON_PATH.relative_to(ROOT_DIR)}")
    print(f"F1 plot: {F1_OVER_TIME_PATH.relative_to(ROOT_DIR)}")
    print(f"Accuracy plot: {ACCURACY_OVER_TIME_PATH.relative_to(ROOT_DIR)}")
    print(f"Average F1 plot: {AVERAGE_F1_PATH.relative_to(ROOT_DIR)}")
    print(f"Update-cost plot: {UPDATE_COST_PATH.relative_to(ROOT_DIR)}")
    if not summary_df.empty:
        print("\nComparison summary:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
