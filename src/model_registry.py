"""Local model registry that simulates cloud model storage."""

from pathlib import Path

import joblib

from src.config import MODEL_STORAGE_DIR


def save_model(model, model_name: str) -> Path:
    """Persist a model artifact to local cloud model storage."""
    MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_STORAGE_DIR / model_name
    joblib.dump(model, path)
    return path


def load_model(model_name: str):
    """Load a model artifact from local cloud model storage."""
    return joblib.load(MODEL_STORAGE_DIR / model_name)
