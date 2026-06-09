"""Run stream simulation with ADWIN drift detection."""

import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.adwin_detector import ADWINDriftDetector, calculate_detection_delay
from src.config import (
    CLOUD_MODEL_DIR,
    FIGURE_DIR,
    METRIC_DIR,
    TRAIN_RATIO,
    WINDOW_SIZE,
)
from src.data_loader import load_synthetic_dataset, time_based_split
from src.evaluation import (
    evaluate_by_windows,
    plot_metric_over_time,
    save_metrics_json,
)
from src.preprocessing import Preprocessor, clean_features, create_binary_label
from src.static_model import load_model


MODEL_PATH = CLOUD_MODEL_DIR / "static_random_forest.joblib"
PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "preprocessor.joblib"
ACTUAL_DRIFT_POINTS_PATH = METRIC_DIR / "synthetic_drift_points.json"
DETECTED_DRIFTS_PATH = METRIC_DIR / "adwin_detected_drifts.json"
DETECTION_DELAY_PATH = METRIC_DIR / "adwin_detection_delay.csv"
ERROR_RATE_PLOT_PATH = FIGURE_DIR / "adwin_error_rate_over_time.png"
F1_PLOT_PATH = FIGURE_DIR / "adwin_f1_over_time.png"


def _relative(path: Path) -> Path:
    """Return a project-relative path for console output."""
    return path.relative_to(ROOT_DIR)


def _require_input_artifacts() -> None:
    """Fail with actionable commands when required artifacts are missing."""
    missing_model_artifacts = [
        path for path in (MODEL_PATH, PREPROCESSOR_PATH) if not path.exists()
    ]
    if missing_model_artifacts:
        missing = ", ".join(str(_relative(path)) for path in missing_model_artifacts)
        raise FileNotFoundError(
            f"Missing static model artifacts: {missing}. "
            "Run: python scripts/02_train_static.py"
        )

    if not ACTUAL_DRIFT_POINTS_PATH.exists():
        raise FileNotFoundError(
            f"Actual drift points file not found: {_relative(ACTUAL_DRIFT_POINTS_PATH)}. "
            "Run: python scripts/00_generate_synthetic_data.py"
        )


def _load_actual_stream_drifts(stream_start: int, stream_end: int) -> list[int]:
    """Load actual drift points that fall inside the evaluated stream."""
    payload = json.loads(ACTUAL_DRIFT_POINTS_PATH.read_text(encoding="utf-8"))
    return [
        int(point)
        for point in payload.get("drift_points", [])
        if stream_start <= int(point) < stream_end
    ]


def main() -> None:
    """Run static-model prediction errors through ADWIN sample by sample."""
    _require_input_artifacts()

    df = create_binary_label(load_synthetic_dataset())
    train_df, stream_df = time_based_split(
        df,
        train_ratio=TRAIN_RATIO,
        timestamp_col="timestamp",
    )

    _, _, feature_names = clean_features(train_df)
    X_stream, y_stream, _ = clean_features(stream_df)
    X_stream = X_stream[feature_names]

    model = load_model(MODEL_PATH)
    preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
    X_stream_scaled = preprocessor.transform(X_stream)

    predictions = model.predict(X_stream_scaled)
    errors = (np.asarray(predictions) != np.asarray(y_stream)).astype(int)

    stream_start = len(train_df)
    stream_end = stream_start + len(stream_df)
    actual_drift_points = _load_actual_stream_drifts(stream_start, stream_end)

    detector = ADWINDriftDetector(delta=0.002)
    for stream_offset, error_value in enumerate(errors):
        absolute_index = stream_start + stream_offset
        if detector.update(int(error_value), absolute_index):
            print(f"ADWIN drift detected at index: {absolute_index}")

    detected_drifts = detector.get_detected_drifts()
    if not detected_drifts:
        print("ADWIN did not detect any drift in the evaluated stream.")

    delay_df = calculate_detection_delay(actual_drift_points, detected_drifts)
    DETECTION_DELAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    delay_df.to_csv(DETECTION_DELAY_PATH, index=False)

    detected_payload = {
        "detector": "ADWIN",
        "delta": detector.delta,
        "stream_start_index": stream_start,
        "stream_end_index": stream_end - 1,
        "stream_samples": len(stream_df),
        "total_prediction_errors": int(errors.sum()),
        "overall_error_rate": float(errors.mean()),
        "actual_drift_points": actual_drift_points,
        "detected_drifts": detected_drifts,
        "number_of_detected_drifts": len(detected_drifts),
    }
    save_metrics_json(detected_payload, DETECTED_DRIFTS_PATH)

    window_metrics_df = evaluate_by_windows(
        model,
        X_stream_scaled,
        y_stream,
        window_size=WINDOW_SIZE,
    )
    window_metrics_df["start_index"] += stream_start
    window_metrics_df["end_index"] += stream_start
    window_metrics_df["error_rate"] = 1.0 - window_metrics_df["accuracy"]

    plot_metric_over_time(
        window_metrics_df,
        metric_name="error_rate",
        output_path=ERROR_RATE_PLOT_PATH,
        drift_points=actual_drift_points,
        detected_drifts=detected_drifts,
    )
    plot_metric_over_time(
        window_metrics_df,
        metric_name="f1",
        output_path=F1_PLOT_PATH,
        drift_points=actual_drift_points,
        detected_drifts=detected_drifts,
    )

    print(f"Actual drift points in stream: {actual_drift_points}")
    print(f"Detected drift points: {detected_drifts}")
    print(f"Detected drifts JSON: {_relative(DETECTED_DRIFTS_PATH)}")
    print(f"Detection delay CSV: {_relative(DETECTION_DELAY_PATH)}")
    print(f"Error-rate plot: {_relative(ERROR_RATE_PLOT_PATH)}")
    print(f"F1 plot: {_relative(F1_PLOT_PATH)}")
    print("\nDetection delay:")
    print(delay_df.to_string(index=False))


if __name__ == "__main__":
    main()
