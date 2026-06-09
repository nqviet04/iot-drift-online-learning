"""Compare a static Random Forest with ADWIN-triggered adaptive retraining."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptive_trainer import (
    AdaptiveRandomForestTrainer,
    RecentBuffer,
    save_retraining_log,
)
from src.adwin_detector import ADWINDriftDetector
from src.config import (
    CLOUD_MODEL_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    RANDOM_STATE,
    RECENT_BUFFER_SIZE,
    TRAIN_RATIO,
    WINDOW_SIZE,
)
from src.data_loader import load_synthetic_dataset, time_based_split
from src.evaluation import (
    compute_classification_metrics,
    save_metrics_json,
    save_window_metrics_csv,
)
from src.preprocessing import Preprocessor, clean_features, create_binary_label
from src.static_model import save_model, train_random_forest


INITIAL_MODEL_PATH = CLOUD_MODEL_DIR / "adaptive_rf_v0.joblib"
WINDOW_METRICS_PATH = METRIC_DIR / "adaptive_static_window_metrics.csv"
RETRAIN_LOG_PATH = METRIC_DIR / "adaptive_static_retrain_log.csv"
RETRAIN_LOG_JSON_PATH = METRIC_DIR / "adaptive_static_retrain_log.json"
SUMMARY_PATH = METRIC_DIR / "adaptive_static_summary.json"
COMPARISON_PLOT_PATH = FIGURE_DIR / "static_vs_adaptive_f1.png"
ACTUAL_DRIFT_POINTS_PATH = METRIC_DIR / "synthetic_drift_points.json"
STREAM_BATCH_SIZE = 256


def _relative(path: str | Path) -> Path:
    """Return a project-relative path for readable logs."""
    return Path(path).resolve().relative_to(ROOT_DIR.resolve())


def _load_actual_stream_drifts(stream_start: int, stream_end: int) -> list[int]:
    """Load actual synthetic drift points inside the evaluated stream."""
    if not ACTUAL_DRIFT_POINTS_PATH.exists():
        return []

    payload = json.loads(ACTUAL_DRIFT_POINTS_PATH.read_text(encoding="utf-8"))
    return [
        int(point)
        for point in payload.get("drift_points", [])
        if stream_start <= int(point) < stream_end
    ]


def _evaluate_prediction_windows(
    y_true: np.ndarray,
    static_predictions: np.ndarray,
    adaptive_predictions: np.ndarray,
    stream_start: int,
) -> pd.DataFrame:
    """Compute static and adaptive metrics from sequential predictions."""
    rows: list[dict[str, float | int]] = []

    for window_id, start in enumerate(range(0, len(y_true), WINDOW_SIZE)):
        end = min(start + WINDOW_SIZE, len(y_true))
        static_metrics = compute_classification_metrics(
            y_true[start:end],
            static_predictions[start:end],
        )
        adaptive_metrics = compute_classification_metrics(
            y_true[start:end],
            adaptive_predictions[start:end],
        )

        rows.append(
            {
                "window_id": window_id,
                "start_index": stream_start + start,
                "end_index": stream_start + end - 1,
                "static_accuracy": static_metrics["accuracy"],
                "static_precision": static_metrics["precision"],
                "static_recall": static_metrics["recall"],
                "static_f1": static_metrics["f1"],
                "adaptive_accuracy": adaptive_metrics["accuracy"],
                "adaptive_precision": adaptive_metrics["precision"],
                "adaptive_recall": adaptive_metrics["recall"],
                "adaptive_f1": adaptive_metrics["f1"],
            }
        )

    return pd.DataFrame(rows)


def _point_to_window_position(window_metrics: pd.DataFrame, point: int) -> float | None:
    """Convert an absolute sample index to its plotted window position."""
    for row in window_metrics.itertuples(index=False):
        if row.start_index <= point <= row.end_index:
            width = max(row.end_index - row.start_index + 1, 1)
            return row.window_id + (point - row.start_index) / width
    return None


def _plot_f1_comparison(
    window_metrics: pd.DataFrame,
    actual_drifts: list[int],
    detected_drifts: list[int],
) -> Path:
    """Plot static and adaptive F1 over stream windows."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    COMPARISON_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        window_metrics["window_id"],
        window_metrics["static_f1"],
        marker="o",
        linewidth=2,
        label="Static Random Forest",
    )
    ax.plot(
        window_metrics["window_id"],
        window_metrics["adaptive_f1"],
        marker="s",
        linewidth=2,
        label="Adaptive Random Forest",
    )

    actual_label_used = False
    for point in actual_drifts:
        position = _point_to_window_position(window_metrics, point)
        if position is not None:
            ax.axvline(
                position,
                color="tab:red",
                linestyle="--",
                alpha=0.8,
                label="Actual drift" if not actual_label_used else None,
            )
            actual_label_used = True

    detected_label_used = False
    for point in detected_drifts:
        position = _point_to_window_position(window_metrics, point)
        if position is not None:
            ax.axvline(
                position,
                color="tab:purple",
                linestyle=":",
                alpha=0.9,
                label="Detected drift" if not detected_label_used else None,
            )
            detected_label_used = True

    ax.set_title("Static vs Adaptive Random Forest F1")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(COMPARISON_PLOT_PATH, dpi=150)
    plt.close(fig)
    return COMPARISON_PLOT_PATH


def main() -> None:
    """Run the static-vs-adaptive stream experiment."""
    df = create_binary_label(load_synthetic_dataset())
    train_df, stream_df = time_based_split(
        df,
        train_ratio=TRAIN_RATIO,
        timestamp_col="timestamp",
    )

    X_train, y_train, feature_names = clean_features(train_df)
    X_stream, y_stream, _ = clean_features(stream_df)
    X_stream = X_stream[feature_names]

    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train).to_numpy()
    X_stream_scaled = preprocessor.transform(X_stream).to_numpy()
    y_train_array = y_train.to_numpy()
    y_stream_array = y_stream.to_numpy()

    print(f"Training initial Random Forest on {len(y_train_array)} samples...")
    initial_model = train_random_forest(
        X_train_scaled,
        y_train_array,
        random_state=RANDOM_STATE,
    )
    save_model(initial_model, INITIAL_MODEL_PATH)

    static_model = copy.deepcopy(initial_model)
    adaptive_trainer = AdaptiveRandomForestTrainer(
        model=initial_model,
        model_version=0,
        random_state=RANDOM_STATE,
    )
    buffer = RecentBuffer(max_size=RECENT_BUFFER_SIZE)
    detector = ADWINDriftDetector(delta=0.002)

    static_predictions = static_model.predict(X_stream_scaled)
    adaptive_predictions: list[int] = []
    retrain_records: list[dict] = []
    stream_start = len(train_df)
    stream_end = stream_start + len(stream_df)

    print(
        f"Processing {len(y_stream_array)} stream samples "
        f"in mini-batches of {STREAM_BATCH_SIZE}..."
    )
    stream_offset = 0
    while stream_offset < len(y_stream_array):
        batch_end = min(stream_offset + STREAM_BATCH_SIZE, len(y_stream_array))
        batch_predictions = adaptive_trainer.model.predict(
            X_stream_scaled[stream_offset:batch_end]
        )
        drift_in_batch = False

        for local_offset, y_pred in enumerate(batch_predictions):
            current_offset = stream_offset + local_offset
            x_row = X_stream_scaled[current_offset]
            y_true = int(y_stream_array[current_offset])
            absolute_index = stream_start + current_offset
            y_pred = int(y_pred)
            adaptive_predictions.append(y_pred)

            error = int(y_pred != y_true)
            buffer.add(x_row, y_true)

            if detector.update(error, absolute_index):
                print(
                    f"[DRIFT] ADWIN detected drift at index {absolute_index}; "
                    f"buffer size={len(buffer)}"
                )
                X_recent, y_recent = buffer.get_data()
                retrain_info = adaptive_trainer.retrain(X_recent, y_recent)
                retrain_record = {
                    "detected_drift_index": absolute_index,
                    **retrain_info,
                }
                retrain_records.append(retrain_record)
                print(
                    f"[RETRAIN] version={retrain_info['version']} "
                    f"samples={retrain_info['n_samples']} "
                    f"time={retrain_info['retrain_time_seconds']:.4f}s "
                    f"model={_relative(retrain_info['model_path'])}"
                )
                stream_offset = current_offset + 1
                drift_in_batch = True
                break

        if not drift_in_batch:
            stream_offset = batch_end

    adaptive_predictions_array = np.asarray(adaptive_predictions)
    detected_drifts = detector.get_detected_drifts()
    actual_drifts = _load_actual_stream_drifts(stream_start, stream_end)

    if not detected_drifts:
        print("No ADWIN drift was detected; adaptive model remained at version 0.")

    window_metrics = _evaluate_prediction_windows(
        y_stream_array,
        np.asarray(static_predictions),
        adaptive_predictions_array,
        stream_start,
    )
    save_window_metrics_csv(window_metrics, WINDOW_METRICS_PATH)
    save_retraining_log(
        retrain_records,
        RETRAIN_LOG_PATH,
        RETRAIN_LOG_JSON_PATH,
    )
    _plot_f1_comparison(window_metrics, actual_drifts, detected_drifts)

    static_metrics = compute_classification_metrics(
        y_stream_array,
        static_predictions,
    )
    adaptive_metrics = compute_classification_metrics(
        y_stream_array,
        adaptive_predictions_array,
    )
    total_retrain_time = sum(
        float(record["retrain_time_seconds"]) for record in retrain_records
    )

    summary = {
        "experiment": "adaptive_static_random_forest",
        "train_rows": len(train_df),
        "stream_rows": len(stream_df),
        "feature_count": len(feature_names),
        "window_size": WINDOW_SIZE,
        "stream_batch_size": STREAM_BATCH_SIZE,
        "recent_buffer_size": RECENT_BUFFER_SIZE,
        "adwin_delta": detector.delta,
        "actual_drift_points": actual_drifts,
        "detected_drift_points": detected_drifts,
        "number_of_retrains": len(retrain_records),
        "final_model_version": adaptive_trainer.model_version,
        "total_retrain_time_seconds": total_retrain_time,
        "average_retrain_time_seconds": (
            total_retrain_time / len(retrain_records) if retrain_records else 0.0
        ),
        "static_metrics": static_metrics,
        "adaptive_metrics": adaptive_metrics,
        "adaptive_minus_static_f1": adaptive_metrics["f1"] - static_metrics["f1"],
        "initial_model_path": str(_relative(INITIAL_MODEL_PATH)),
        "window_metrics_path": str(_relative(WINDOW_METRICS_PATH)),
        "retrain_log_path": str(_relative(RETRAIN_LOG_PATH)),
        "comparison_plot_path": str(_relative(COMPARISON_PLOT_PATH)),
    }
    save_metrics_json(summary, SUMMARY_PATH)

    print("\nAdaptive static experiment completed.")
    print(f"Actual drift points: {actual_drifts}")
    print(f"Detected drift points: {detected_drifts}")
    print(f"Retrain count: {len(retrain_records)}")
    print(f"Final model version: {adaptive_trainer.model_version}")
    print(f"Static F1: {static_metrics['f1']:.4f}")
    print(f"Adaptive F1: {adaptive_metrics['f1']:.4f}")
    print(f"Window metrics: {_relative(WINDOW_METRICS_PATH)}")
    print(f"Retrain log: {_relative(RETRAIN_LOG_PATH)}")
    print(f"Comparison plot: {_relative(COMPARISON_PLOT_PATH)}")
    print(f"Summary: {_relative(SUMMARY_PATH)}")


if __name__ == "__main__":
    main()
