"""Train and evaluate the initial LSTM on the synthetic IoT stream."""

import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLOUD_MODEL_DIR,
    FIGURE_DIR,
    LSTM_TIMESTEPS,
    METRIC_DIR,
    RANDOM_STATE,
    TRAIN_RATIO,
)
from src.data_loader import load_synthetic_dataset, time_based_split
from src.evaluation import compute_classification_metrics, save_metrics_json
from src.lstm_model import (
    build_lstm_model,
    create_sequences,
    predict_lstm_binary,
    save_lstm_model,
    train_lstm_model,
)
from src.preprocessing import Preprocessor, clean_features, create_binary_label


MODEL_PATH = CLOUD_MODEL_DIR / "lstm_initial.keras"
PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "lstm_preprocessor.joblib"
METRICS_PATH = METRIC_DIR / "lstm_initial_metrics.json"
HISTORY_PLOT_PATH = FIGURE_DIR / "lstm_training_history.png"
VALIDATION_RATIO = 0.2
EPOCHS = 5
BATCH_SIZE = 64


def _relative(path: str | Path) -> Path:
    """Return a project-relative path for console output."""
    return Path(path).resolve().relative_to(ROOT_DIR.resolve())


def _plot_training_history(history, output_path: Path) -> Path:
    """Save training and validation loss/accuracy curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for training-history plots. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    history_values = history.history
    epochs = range(1, len(history_values["loss"]) + 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history_values["loss"], marker="o", label="Train")
    if "val_loss" in history_values:
        axes[0].plot(epochs, history_values["val_loss"], marker="s", label="Validation")
    axes[0].set_title("LSTM Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history_values["accuracy"], marker="o", label="Train")
    if "val_accuracy" in history_values:
        axes[1].plot(
            epochs,
            history_values["val_accuracy"],
            marker="s",
            label="Validation",
        )
    axes[1].set_title("LSTM Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    """Train the initial time-series LSTM and evaluate it on stream data."""
    import tensorflow as tf

    np.random.seed(RANDOM_STATE)
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    df = create_binary_label(load_synthetic_dataset())
    initial_train_df, stream_df = time_based_split(
        df,
        train_ratio=TRAIN_RATIO,
        timestamp_col="timestamp",
    )

    validation_start = int(len(initial_train_df) * (1.0 - VALIDATION_RATIO))
    fit_df = initial_train_df.iloc[:validation_start].copy()
    validation_df = initial_train_df.iloc[validation_start:].copy()

    X_fit, y_fit, feature_names = clean_features(fit_df)
    X_validation, y_validation, _ = clean_features(validation_df)
    X_stream, y_stream, _ = clean_features(stream_df)
    X_validation = X_validation[feature_names]
    X_stream = X_stream[feature_names]

    preprocessor = Preprocessor()
    X_fit_scaled = preprocessor.fit_transform(X_fit)
    X_validation_scaled = preprocessor.transform(X_validation)
    X_stream_scaled = preprocessor.transform(X_stream)

    X_fit_seq, y_fit_seq = create_sequences(
        X_fit_scaled,
        y_fit,
        timesteps=LSTM_TIMESTEPS,
    )
    X_validation_seq, y_validation_seq = create_sequences(
        X_validation_scaled,
        y_validation,
        timesteps=LSTM_TIMESTEPS,
    )
    X_stream_seq, y_stream_seq = create_sequences(
        X_stream_scaled,
        y_stream,
        timesteps=LSTM_TIMESTEPS,
    )

    print(f"Training sequences: {X_fit_seq.shape}")
    print(f"Validation sequences: {X_validation_seq.shape}")
    print(f"Stream sequences: {X_stream_seq.shape}")

    model = build_lstm_model(
        input_shape=(LSTM_TIMESTEPS, len(feature_names))
    )
    model, history = train_lstm_model(
        model,
        X_fit_seq,
        y_fit_seq,
        X_val_seq=X_validation_seq,
        y_val_seq=y_validation_seq,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    stream_predictions = predict_lstm_binary(model, X_stream_seq)
    stream_metrics = compute_classification_metrics(
        y_stream_seq,
        stream_predictions,
    )

    saved_model_path = save_lstm_model(model, MODEL_PATH)
    saved_preprocessor_path = preprocessor.save(PREPROCESSOR_PATH)
    saved_history_path = _plot_training_history(history, HISTORY_PLOT_PATH)

    metrics_payload = {
        "model": "LSTM",
        "architecture": [
            "LSTM(64)",
            "Dropout(0.3)",
            "Dense(32, relu)",
            "Dropout(0.2)",
            "Dense(1, sigmoid)",
        ],
        "train_rows_before_sequence": len(fit_df),
        "validation_rows_before_sequence": len(validation_df),
        "stream_rows_before_sequence": len(stream_df),
        "train_sequences": len(X_fit_seq),
        "validation_sequences": len(X_validation_seq),
        "stream_sequences": len(X_stream_seq),
        "timesteps": LSTM_TIMESTEPS,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "epochs_requested": EPOCHS,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "stream_metrics": stream_metrics,
        "model_path": str(_relative(saved_model_path)),
        "preprocessor_path": str(_relative(saved_preprocessor_path)),
        "training_history_plot_path": str(_relative(saved_history_path)),
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    save_metrics_json(metrics_payload, METRICS_PATH)

    print("\nInitial LSTM training completed.")
    print(f"Model saved to: {_relative(saved_model_path)}")
    print(f"Preprocessor saved to: {_relative(saved_preprocessor_path)}")
    print(f"Metrics saved to: {_relative(METRICS_PATH)}")
    print(f"Training history plot: {_relative(saved_history_path)}")
    print(f"Confusion matrix: {stream_metrics['confusion_matrix']}")
    print(f"Accuracy:  {stream_metrics['accuracy']:.4f}")
    print(f"Precision: {stream_metrics['precision']:.4f}")
    print(f"Recall:    {stream_metrics['recall']:.4f}")
    print(f"F1-score:  {stream_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
