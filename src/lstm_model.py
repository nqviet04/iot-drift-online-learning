"""LSTM sequence creation, training, prediction, and persistence helpers."""

from pathlib import Path
from typing import Any

import numpy as np


def _get_tensorflow():
    """Import TensorFlow with an actionable dependency error."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "tensorflow is required for LSTM models. "
            "Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return tf


def create_sequences(
    X,
    y,
    timesteps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ordered tabular rows into overlapping time sequences.

    The label for each sequence is the label at the final timestep.
    """
    if timesteps <= 0:
        raise ValueError("timesteps must be greater than 0.")

    X_array = np.asarray(X, dtype=np.float32)
    y_array = np.asarray(y).reshape(-1)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array with shape (samples, features).")
    if len(X_array) != len(y_array):
        raise ValueError("X and y must contain the same number of samples.")
    if len(X_array) < timesteps:
        raise ValueError(
            f"Not enough samples to create sequences: got {len(X_array)}, "
            f"need at least {timesteps}."
        )

    n_sequences = len(X_array) - timesteps + 1
    X_seq = np.empty(
        (n_sequences, timesteps, X_array.shape[1]),
        dtype=np.float32,
    )
    for sequence_index in range(n_sequences):
        X_seq[sequence_index] = X_array[
            sequence_index : sequence_index + timesteps
        ]

    y_seq = y_array[timesteps - 1 :].astype(np.int32, copy=False)
    return X_seq, y_seq


def build_lstm_model(input_shape: tuple[int, int]):
    """Build and compile the initial binary-classification LSTM."""
    if len(input_shape) != 2 or any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape must be (timesteps, n_features).")

    tf = _get_tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="iot_drift_lstm",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm_model(
    model,
    X_train_seq,
    y_train_seq,
    X_val_seq=None,
    y_val_seq=None,
    epochs: int = 5,
    batch_size: int = 64,
) -> tuple[Any, Any]:
    """Train an LSTM, using early stopping when validation data is supplied."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if len(X_train_seq) != len(y_train_seq):
        raise ValueError("X_train_seq and y_train_seq must have equal lengths.")
    if len(X_train_seq) == 0:
        raise ValueError("Training sequences cannot be empty.")

    has_validation = X_val_seq is not None or y_val_seq is not None
    if has_validation and (X_val_seq is None or y_val_seq is None):
        raise ValueError("Provide both X_val_seq and y_val_seq, or neither.")
    if has_validation and len(X_val_seq) != len(y_val_seq):
        raise ValueError("X_val_seq and y_val_seq must have equal lengths.")

    callbacks = []
    validation_data = None
    if has_validation:
        tf = _get_tensorflow()
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True,
            )
        )
        validation_data = (X_val_seq, y_val_seq)

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    return model, history


def predict_lstm_binary(
    model,
    X_seq,
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict binary labels from sigmoid probabilities."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    if len(X_seq) == 0:
        return np.empty(0, dtype=np.int32)

    probabilities = np.asarray(model.predict(X_seq, verbose=0)).reshape(-1)
    return (probabilities >= threshold).astype(np.int32)


def save_lstm_model(model, path: str | Path) -> Path:
    """Save a Keras model to disk."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    return output_path


def load_lstm_model(path: str | Path):
    """Load a saved Keras model."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"LSTM model file not found: {input_path}")

    tf = _get_tensorflow()
    return tf.keras.models.load_model(input_path)
