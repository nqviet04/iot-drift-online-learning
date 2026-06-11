"""Run ADWIN-triggered online fine-tuning for the initial LSTM model."""

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adaptive_trainer import RecentBuffer, save_retraining_log
from src.adwin_detector import ADWINDriftDetector
from src.config import (
    CLOUD_MODEL_DIR,
    FIGURE_DIR,
    LSTM_TIMESTEPS,
    METRIC_DIR,
    RECENT_BUFFER_SIZE,
    TRAIN_RATIO,
    WINDOW_SIZE,
)
from src.data_loader import load_synthetic_dataset, time_based_split
from src.evaluation import (
    compute_classification_metrics,
    plot_metric_over_time,
    save_metrics_json,
    save_window_metrics_csv,
)
from src.lstm_model import (
    create_sequences,
    load_lstm_model,
    predict_lstm_binary,
    save_lstm_model,
)
from src.preprocessing import Preprocessor, clean_features, create_binary_label


INITIAL_MODEL_PATH = CLOUD_MODEL_DIR / "lstm_initial.keras"
PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "lstm_preprocessor.joblib"
ACTUAL_DRIFT_POINTS_PATH = METRIC_DIR / "synthetic_drift_points.json"

WINDOW_METRICS_PATH = METRIC_DIR / "adaptive_lstm_window_metrics.csv"
RETRAIN_LOG_PATH = METRIC_DIR / "adaptive_lstm_retrain_log.csv"
DETECTED_DRIFTS_PATH = METRIC_DIR / "adaptive_lstm_detected_drifts.json"
ADAPTIVE_F1_PLOT_PATH = FIGURE_DIR / "adaptive_lstm_f1_over_time.png"
COMPARISON_PLOT_PATH = FIGURE_DIR / "lstm_static_vs_adaptive.png"
SUMMARY_PATH = METRIC_DIR / "adaptive_lstm_summary.json"

ADWIN_DELTA = 0.002
FINE_TUNE_EPOCHS = 3
FINE_TUNE_BATCH_SIZE = 64


def _relative(path: str | Path) -> Path:
    """Return a project-relative path for readable logs."""
    return Path(path).resolve().relative_to(ROOT_DIR.resolve())


def _require_initial_artifacts() -> None:
    """Require the model and preprocessor produced by script 05."""
    missing = [
        path
        for path in (INITIAL_MODEL_PATH, PREPROCESSOR_PATH)
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(_relative(path)) for path in missing)
        raise FileNotFoundError(
            f"Missing initial LSTM artifacts: {missing_text}. "
            "Run: python scripts/05_train_lstm.py"
        )


def _load_actual_stream_drifts(stream_start: int, stream_end: int) -> list[int]:
    """Load true synthetic drift points inside the evaluated stream."""
    if not ACTUAL_DRIFT_POINTS_PATH.exists():
        return []

    payload = json.loads(ACTUAL_DRIFT_POINTS_PATH.read_text(encoding="utf-8"))
    return [
        int(point)
        for point in payload.get("drift_points", [])
        if stream_start <= int(point) < stream_end
    ]


def _plot_static_vs_adaptive(
    window_metrics: pd.DataFrame,
    actual_drifts: list[int],
    detected_drifts: list[int],
) -> Path:
    """Plot static and adaptive LSTM F1 over stream windows."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    def point_to_window(point: int) -> float | None:
        for row in window_metrics.itertuples(index=False):
            if row.start_index <= point <= row.end_index:
                width = max(row.end_index - row.start_index + 1, 1)
                return row.window_id + (point - row.start_index) / width
        return None

    COMPARISON_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        window_metrics["window_id"],
        window_metrics["static_f1"],
        marker="o",
        linewidth=2,
        label="Static LSTM",
    )
    ax.plot(
        window_metrics["window_id"],
        window_metrics["adaptive_f1"],
        marker="s",
        linewidth=2,
        label="Adaptive LSTM",
    )

    actual_label_used = False
    for point in actual_drifts:
        position = point_to_window(point)
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
        position = point_to_window(point)
        if position is not None:
            ax.axvline(
                position,
                color="tab:purple",
                linestyle=":",
                alpha=0.9,
                label="Detected drift" if not detected_label_used else None,
            )
            detected_label_used = True

    ax.set_title("Static vs Adaptive LSTM F1")
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
    """Evaluate LSTM windows and fine-tune after ADWIN drift detections."""
    _require_initial_artifacts()

    df = create_binary_label(load_synthetic_dataset())
    train_df, stream_df = time_based_split(
        df,
        train_ratio=TRAIN_RATIO,
        timestamp_col="timestamp",
    )

    _, _, train_feature_names = clean_features(train_df)
    X_stream, y_stream, _ = clean_features(stream_df)

    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
    feature_names = preprocessor.feature_names or train_feature_names
    X_stream = X_stream[feature_names]
    X_stream_scaled = preprocessor.transform(X_stream).to_numpy(dtype=np.float32)
    y_stream_array = y_stream.to_numpy(dtype=np.int32)

    static_model = load_lstm_model(INITIAL_MODEL_PATH)
    adaptive_model = load_lstm_model(INITIAL_MODEL_PATH)
    detector = ADWINDriftDetector(delta=ADWIN_DELTA)
    recent_buffer = RecentBuffer(max_size=RECENT_BUFFER_SIZE)

    stream_start = len(train_df)
    stream_end = stream_start + len(stream_df)
    actual_drifts = _load_actual_stream_drifts(stream_start, stream_end)

    window_records: list[dict] = []
    retrain_records: list[dict] = []
    all_static_true: list[np.ndarray] = []
    all_static_pred: list[np.ndarray] = []
    all_adaptive_true: list[np.ndarray] = []
    all_adaptive_pred: list[np.ndarray] = []
    model_version = 0
    sequence_context_X = np.empty(
        (0, X_stream_scaled.shape[1]),
        dtype=np.float32,
    )
    sequence_context_y = np.empty(0, dtype=np.int32)

    total_windows = (len(X_stream_scaled) + WINDOW_SIZE - 1) // WINDOW_SIZE
    print(
        f"Processing {len(X_stream_scaled)} stream rows in "
        f"{total_windows} windows..."
    )

    for window_id, window_start in enumerate(
        range(0, len(X_stream_scaled), WINDOW_SIZE)
    ):
        window_end = min(window_start + WINDOW_SIZE, len(X_stream_scaled))
        X_window = X_stream_scaled[window_start:window_end]
        y_window = y_stream_array[window_start:window_end]
        context_rows = len(sequence_context_X)

        if context_rows + len(X_window) < LSTM_TIMESTEPS:
            print(f"Skipping short window {window_id}: {len(X_window)} rows.")
            continue

        X_sequence_input = np.concatenate(
            [sequence_context_X, X_window],
            axis=0,
        )
        y_sequence_input = np.concatenate(
            [sequence_context_y, y_window],
            axis=0,
        )
        X_window_seq, y_window_seq = create_sequences(
            X_sequence_input,
            y_sequence_input,
            timesteps=LSTM_TIMESTEPS,
        )
        context_size = LSTM_TIMESTEPS - 1
        sequence_context_X = X_sequence_input[-context_size:].copy()
        sequence_context_y = y_sequence_input[-context_size:].copy()
        static_predictions = predict_lstm_binary(static_model, X_window_seq)
        adaptive_predictions = predict_lstm_binary(adaptive_model, X_window_seq)

        static_metrics = compute_classification_metrics(
            y_window_seq,
            static_predictions,
        )
        adaptive_metrics = compute_classification_metrics(
            y_window_seq,
            adaptive_predictions,
        )
        version_used = model_version

        all_static_true.append(y_window_seq)
        all_static_pred.append(static_predictions)
        all_adaptive_true.append(y_window_seq)
        all_adaptive_pred.append(adaptive_predictions)

        # Labels become available at the end of the current stream window.
        for x_row, y_value in zip(X_window, y_window):
            recent_buffer.add(x_row, int(y_value))

        window_detected_drifts: list[int] = []
        sequence_start_absolute = (
            stream_start
            + window_start
            + max(LSTM_TIMESTEPS - 1 - context_rows, 0)
        )
        for sequence_offset, (y_true, y_pred) in enumerate(
            zip(y_window_seq, adaptive_predictions)
        ):
            absolute_index = sequence_start_absolute + sequence_offset
            error = int(int(y_true) != int(y_pred))
            if detector.update(error, absolute_index):
                window_detected_drifts.append(absolute_index)
                print(
                    f"[DRIFT] window={window_id} "
                    f"detected_index={absolute_index}"
                )

        if window_detected_drifts:
            X_recent, y_recent = recent_buffer.get_data()
            X_recent_seq, y_recent_seq = create_sequences(
                X_recent,
                y_recent,
                timesteps=LSTM_TIMESTEPS,
            )

            start_time = perf_counter()
            adaptive_model.fit(
                X_recent_seq,
                y_recent_seq,
                epochs=FINE_TUNE_EPOCHS,
                batch_size=FINE_TUNE_BATCH_SIZE,
                shuffle=False,
                verbose=0,
            )
            retrain_time = perf_counter() - start_time
            model_version += 1
            model_path = (
                CLOUD_MODEL_DIR / f"lstm_adaptive_v{model_version}.keras"
            )
            save_lstm_model(adaptive_model, model_path)

            retrain_record = {
                "window_id": window_id,
                "detected_drift_index": window_detected_drifts[0],
                "version": model_version,
                "retrain_time_seconds": retrain_time,
                "n_buffer_rows": len(y_recent),
                "n_sequences": len(y_recent_seq),
                "epochs": FINE_TUNE_EPOCHS,
                "batch_size": FINE_TUNE_BATCH_SIZE,
                "model_path": str(_relative(model_path)),
            }
            retrain_records.append(retrain_record)
            print(
                f"[FINE-TUNE] version={model_version} "
                f"rows={len(y_recent)} sequences={len(y_recent_seq)} "
                f"time={retrain_time:.4f}s "
                f"model={_relative(model_path)}"
            )

        window_records.append(
            {
                "window_id": window_id,
                "start_index": stream_start + window_start,
                "end_index": stream_start + window_end - 1,
                "n_rows": len(X_window),
                "n_sequences": len(y_window_seq),
                "context_rows": context_rows,
                "model_version_used": version_used,
                "drift_detected": bool(window_detected_drifts),
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
        print(
            f"Window {window_id + 1}/{total_windows}: "
            f"static_f1={static_metrics['f1']:.4f}, "
            f"adaptive_f1={adaptive_metrics['f1']:.4f}, "
            f"version={version_used}"
        )

    window_metrics = pd.DataFrame(window_records)
    if window_metrics.empty:
        raise RuntimeError("No stream window was large enough to create LSTM sequences.")

    detected_drifts = detector.get_detected_drifts()
    save_window_metrics_csv(window_metrics, WINDOW_METRICS_PATH)
    save_retraining_log(retrain_records, RETRAIN_LOG_PATH)

    detected_payload = {
        "detector": "ADWIN",
        "delta": ADWIN_DELTA,
        "actual_drift_points": actual_drifts,
        "detected_drift_points": detected_drifts,
        "detected_drift_count": len(detected_drifts),
    }
    save_metrics_json(detected_payload, DETECTED_DRIFTS_PATH)

    plot_metric_over_time(
        window_metrics,
        metric_name="adaptive_f1",
        output_path=ADAPTIVE_F1_PLOT_PATH,
        drift_points=actual_drifts,
        detected_drifts=detected_drifts,
    )
    _plot_static_vs_adaptive(
        window_metrics,
        actual_drifts,
        detected_drifts,
    )

    static_overall_metrics = compute_classification_metrics(
        np.concatenate(all_static_true),
        np.concatenate(all_static_pred),
    )
    adaptive_overall_metrics = compute_classification_metrics(
        np.concatenate(all_adaptive_true),
        np.concatenate(all_adaptive_pred),
    )
    retrain_times = [
        float(record["retrain_time_seconds"]) for record in retrain_records
    ]
    adaptive_f1_values = window_metrics["adaptive_f1"]

    summary = {
        "total_windows": len(window_metrics),
        "detected_drift_count": len(detected_drifts),
        "detected_drift_points": detected_drifts,
        "actual_drift_points": actual_drifts,
        "total_retrain_count": len(retrain_records),
        "average_retrain_time_seconds": (
            float(np.mean(retrain_times)) if retrain_times else 0.0
        ),
        "total_retrain_time_seconds": float(sum(retrain_times)),
        "final_model_version": model_version,
        "final_f1": float(adaptive_f1_values.iloc[-1]),
        "best_f1": float(adaptive_f1_values.max()),
        "worst_f1": float(adaptive_f1_values.min()),
        "static_overall_metrics": static_overall_metrics,
        "adaptive_overall_metrics": adaptive_overall_metrics,
        "window_size": WINDOW_SIZE,
        "recent_buffer_size": RECENT_BUFFER_SIZE,
        "timesteps": LSTM_TIMESTEPS,
        "sequence_context_across_windows": True,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "fine_tune_batch_size": FINE_TUNE_BATCH_SIZE,
        "window_metrics_path": str(_relative(WINDOW_METRICS_PATH)),
        "retrain_log_path": str(_relative(RETRAIN_LOG_PATH)),
        "detected_drifts_path": str(_relative(DETECTED_DRIFTS_PATH)),
        "adaptive_f1_plot_path": str(_relative(ADAPTIVE_F1_PLOT_PATH)),
        "comparison_plot_path": str(_relative(COMPARISON_PLOT_PATH)),
    }
    save_metrics_json(summary, SUMMARY_PATH)

    if not detected_drifts:
        print("No ADWIN drift was detected; no LSTM fine-tuning was performed.")

    print("\nAdaptive LSTM experiment completed.")
    print(f"Detected drift points: {detected_drifts}")
    print(f"Retrain count: {len(retrain_records)}")
    print(f"Final model version: {model_version}")
    print(f"Final window F1: {summary['final_f1']:.4f}")
    print(f"Best window F1: {summary['best_f1']:.4f}")
    print(f"Worst window F1: {summary['worst_f1']:.4f}")
    print(f"Window metrics: {_relative(WINDOW_METRICS_PATH)}")
    print(f"Retrain log: {_relative(RETRAIN_LOG_PATH)}")
    print(f"Detected drifts: {_relative(DETECTED_DRIFTS_PATH)}")
    print(f"Adaptive F1 plot: {_relative(ADAPTIVE_F1_PLOT_PATH)}")
    print(f"Comparison plot: {_relative(COMPARISON_PLOT_PATH)}")
    print(f"Summary: {_relative(SUMMARY_PATH)}")


if __name__ == "__main__":
    main()
