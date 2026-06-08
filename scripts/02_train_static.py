"""Train the static baseline model."""

import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CLOUD_MODEL_DIR, METRIC_DIR, RANDOM_STATE
from src.data_loader import load_synthetic_dataset, time_based_split
from src.preprocessing import Preprocessor, clean_features, create_binary_label
from src.static_model import save_model, train_random_forest


MODEL_PATH = CLOUD_MODEL_DIR / "static_random_forest.joblib"
PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = METRIC_DIR / "static_model_metrics.json"


def _evaluate(y_true, y_pred) -> dict:
    """Compute binary metrics and confusion matrix."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _print_metrics(title: str, metrics: dict) -> None:
    """Print a compact metrics block."""
    print(f"\n{title}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")


def _relative(path: Path) -> Path:
    """Return project-relative paths for console output."""
    return path.relative_to(ROOT_DIR)


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

    saved_model_path = save_model(model, MODEL_PATH)
    saved_preprocessor_path = preprocessor.save(PREPROCESSOR_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "model": "RandomForestClassifier",
        "train_rows": len(train_df),
        "stream_rows": len(stream_df),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "train_metrics": train_metrics,
        "stream_metrics": stream_metrics,
        "model_path": str(_relative(saved_model_path)),
        "preprocessor_path": str(_relative(saved_preprocessor_path)),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Static Random Forest training completed.")
    print(f"Model saved to: {_relative(saved_model_path)}")
    print(f"Preprocessor saved to: {_relative(saved_preprocessor_path)}")
    print(f"Metrics saved to: {_relative(METRICS_PATH)}")
    _print_metrics("Train metrics", train_metrics)
    _print_metrics("Stream test metrics", stream_metrics)


if __name__ == "__main__":
    main()
