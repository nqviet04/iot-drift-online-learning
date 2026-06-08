"""Train the static baseline model."""

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CLOUD_MODEL_DIR, FIGURE_DIR, METRIC_DIR, RANDOM_STATE, WINDOW_SIZE
from src.data_loader import load_synthetic_dataset, time_based_split
from src.evaluation import (
    compute_classification_metrics,
    evaluate_by_windows,
    plot_metric_over_time,
    save_metrics_json,
    save_window_metrics_csv,
)
from src.preprocessing import Preprocessor, clean_features, create_binary_label
from src.static_model import save_model, train_random_forest


MODEL_PATH = CLOUD_MODEL_DIR / "static_random_forest.joblib"
PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = METRIC_DIR / "static_model_metrics.json"
WINDOW_METRICS_PATH = METRIC_DIR / "static_window_metrics.csv"
F1_PLOT_PATH = FIGURE_DIR / "static_f1_over_time.png"
SYNTHETIC_DRIFT_POINTS_PATH = METRIC_DIR / "synthetic_drift_points.json"


def _evaluate(y_true, y_pred) -> dict:
    """Compute binary metrics and confusion matrix."""
    return compute_classification_metrics(y_true, y_pred)


def _print_metrics(title: str, metrics: dict) -> None:
    """Print a compact metrics block."""
    print(f"\n{title}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")


def _relative(path: Path) -> Path:
    """Return project-relative paths for console output."""
    return path.relative_to(ROOT_DIR)


def _load_stream_drift_points(train_rows: int, stream_rows: int) -> list[int]:
    """Load true synthetic drift points and convert them to stream-relative rows."""
    if not SYNTHETIC_DRIFT_POINTS_PATH.exists():
        return []

    payload = json.loads(SYNTHETIC_DRIFT_POINTS_PATH.read_text(encoding="utf-8"))
    drift_points = payload.get("drift_points", [])
    stream_end = train_rows + stream_rows
    return [
        int(point) - train_rows
        for point in drift_points
        if train_rows <= int(point) < stream_end
    ]


def main() -> None:
    """Train and evaluate the static Random Forest baseline."""
    df = load_synthetic_dataset()
    df = create_binary_label(df)

    train_df, stream_df = time_based_split(df, train_ratio=0.6, timestamp_col="timestamp")

    X_train, y_train, feature_names = clean_features(train_df)
    X_stream, y_stream, _ = clean_features(stream_df)
    X_stream = X_stream[feature_names]

    preprocessor = Preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_stream_scaled = preprocessor.transform(X_stream)

    model = train_random_forest(X_train_scaled, y_train, random_state=RANDOM_STATE)

    train_pred = model.predict(X_train_scaled)
    stream_pred = model.predict(X_stream_scaled)

    train_metrics = _evaluate(y_train, train_pred)
    stream_metrics = _evaluate(y_stream, stream_pred)
    window_metrics_df = evaluate_by_windows(
        model,
        X_stream_scaled,
        y_stream,
        window_size=WINDOW_SIZE,
    )
    stream_drift_points = _load_stream_drift_points(len(train_df), len(stream_df))

    saved_model_path = save_model(model, MODEL_PATH)
    saved_preprocessor_path = preprocessor.save(PREPROCESSOR_PATH)
    saved_window_metrics_path = save_window_metrics_csv(window_metrics_df, WINDOW_METRICS_PATH)
    saved_f1_plot_path = plot_metric_over_time(
        window_metrics_df,
        metric_name="f1",
        output_path=F1_PLOT_PATH,
        drift_points=stream_drift_points,
    )

    metrics_payload = {
        "model": "RandomForestClassifier",
        "train_rows": len(train_df),
        "stream_rows": len(stream_df),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "window_size": WINDOW_SIZE,
        "stream_relative_drift_points": stream_drift_points,
        "train_metrics": train_metrics,
        "stream_metrics": stream_metrics,
        "model_path": str(_relative(saved_model_path)),
        "preprocessor_path": str(_relative(saved_preprocessor_path)),
        "window_metrics_path": str(_relative(saved_window_metrics_path)),
        "f1_plot_path": str(_relative(saved_f1_plot_path)),
    }
    saved_metrics_path = save_metrics_json(metrics_payload, METRICS_PATH)

    print("Static Random Forest training completed.")
    print(f"Model saved to: {_relative(saved_model_path)}")
    print(f"Preprocessor saved to: {_relative(saved_preprocessor_path)}")
    print(f"Metrics saved to: {_relative(saved_metrics_path)}")
    print(f"Window metrics saved to: {_relative(saved_window_metrics_path)}")
    print(f"F1 plot saved to: {_relative(saved_f1_plot_path)}")
    _print_metrics("Train metrics", train_metrics)
    _print_metrics("Stream test metrics", stream_metrics)


if __name__ == "__main__":
    main()
