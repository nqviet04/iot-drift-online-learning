"""FastAPI endpoint that simulates cloud-hosted IoT model inference.

Run locally with:
    uvicorn api.main:app --reload
"""

import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, RootModel

from src.config import CLOUD_MODEL_DIR, LSTM_TIMESTEPS
from src.lstm_model import load_lstm_model
from src.preprocessing import Preprocessor
from src.static_model import load_model


ADAPTIVE_LSTM_PATTERN = re.compile(r"^lstm_adaptive_v(\d+)\.keras$")
ADAPTIVE_RF_PATTERN = re.compile(r"^adaptive_rf_v(\d+)\.joblib$")
STATIC_RF_FILENAME = "static_random_forest.joblib"
LSTM_PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "lstm_preprocessor.joblib"
RF_PREPROCESSOR_PATH = CLOUD_MODEL_DIR / "preprocessor.joblib"
ADAPTIVE_RF_PREPROCESSOR_PATH = (
    CLOUD_MODEL_DIR / "adaptive_rf_preprocessor.joblib"
)


class FeatureRecord(RootModel[dict[str, float]]):
    """One IoT feature dictionary, for example feature_0 through feature_19."""


class FeatureBatch(RootModel[list[dict[str, float]]]):
    """A batch of IoT feature dictionaries in stream order."""


class PredictionResponse(BaseModel):
    """Prediction returned by the active cloud model."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: int = Field(ge=0, le=1)
    label: str
    model_name: str
    model_version: int | None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionResponse]


class ModelInfo(BaseModel):
    """Metadata for one model artifact in local cloud storage."""

    model_config = ConfigDict(protected_namespaces=())

    filename: str
    model_type: str
    version: int | None
    size_bytes: int
    is_active: bool


@dataclass
class LoadedModel:
    """In-memory model bundle selected at API startup."""

    model: Any
    preprocessor: Preprocessor
    model_name: str
    model_type: str
    version: int | None
    path: Path


@dataclass
class APIState:
    """Mutable API inference state."""

    loaded: LoadedModel | None = None
    load_error: str | None = None


state = APIState()


def _versioned_models(pattern: re.Pattern[str]) -> list[tuple[int, Path]]:
    """Find matching versioned artifacts ordered from newest to oldest."""
    matches: list[tuple[int, Path]] = []
    for path in CLOUD_MODEL_DIR.glob("*"):
        match = pattern.match(path.name)
        if path.is_file() and match:
            matches.append((int(match.group(1)), path))
    return sorted(matches, key=lambda item: item[0], reverse=True)


def _select_model_artifact() -> tuple[str, int | None, Path, Path]:
    """Select the highest-priority model and its matching preprocessor."""
    adaptive_lstm = _versioned_models(ADAPTIVE_LSTM_PATTERN)
    if adaptive_lstm:
        version, path = adaptive_lstm[0]
        return "adaptive_lstm", version, path, LSTM_PREPROCESSOR_PATH

    adaptive_rf = _versioned_models(ADAPTIVE_RF_PATTERN)
    if adaptive_rf:
        version, path = adaptive_rf[0]
        preprocessor_path = (
            ADAPTIVE_RF_PREPROCESSOR_PATH
            if ADAPTIVE_RF_PREPROCESSOR_PATH.exists()
            else RF_PREPROCESSOR_PATH
        )
        return "adaptive_random_forest", version, path, preprocessor_path

    static_path = CLOUD_MODEL_DIR / STATIC_RF_FILENAME
    if static_path.exists():
        return "static_random_forest", None, static_path, RF_PREPROCESSOR_PATH

    raise FileNotFoundError(
        "No prediction model found in cloud_model_storage/. "
        "Run scripts/02_train_static.py or scripts/05_train_lstm.py and "
        "scripts/06_run_adaptive_lstm.py first."
    )


def _load_latest_model() -> LoadedModel:
    """Load the latest model using the configured priority order."""
    model_type, version, model_path, preprocessor_path = _select_model_artifact()
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Required preprocessor not found: {preprocessor_path.name}"
        )

    if model_type == "adaptive_lstm":
        model = load_lstm_model(model_path)
        model_name = "Adaptive LSTM"
    else:
        model = load_model(model_path)
        model_name = (
            "Adaptive Random Forest"
            if model_type == "adaptive_random_forest"
            else "Static Random Forest"
        )

    preprocessor = Preprocessor.load(preprocessor_path)
    if not preprocessor.feature_names:
        raise ValueError(
            f"Preprocessor has no feature metadata: {preprocessor_path.name}"
        )

    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name=model_name,
        model_type=model_type,
        version=version,
        path=model_path,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the latest model once when the API process starts."""
    try:
        state.loaded = _load_latest_model()
        state.load_error = None
        print(
            f"Loaded {state.loaded.model_name} "
            f"version={state.loaded.version} from {state.loaded.path.name}"
        )
    except Exception as exc:  # Keep health/models endpoints available.
        state.loaded = None
        state.load_error = str(exc)
        print(f"Model loading failed: {exc}")
    yield


app = FastAPI(
    title="IoT Drift Online Learning API",
    version="1.0.0",
    description="Local cloud-model endpoint for benign/attack classification.",
    lifespan=lifespan,
)


def _require_loaded_model() -> LoadedModel:
    """Return the active model or a service-unavailable API error."""
    if state.loaded is None:
        detail = state.load_error or "No model is currently loaded."
        raise HTTPException(status_code=503, detail=detail)
    return state.loaded


def _prepare_features(
    records: list[dict[str, float]],
    loaded: LoadedModel,
) -> np.ndarray:
    """Validate, order, and scale request features."""
    if not records:
        raise HTTPException(
            status_code=422,
            detail="At least one feature record is required.",
        )

    feature_names = loaded.preprocessor.feature_names or []
    missing_by_record: dict[int, list[str]] = {}
    for index, record in enumerate(records):
        missing = [name for name in feature_names if name not in record]
        if missing:
            missing_by_record[index] = missing

    if missing_by_record:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request records are missing required features.",
                "missing_features": missing_by_record,
                "expected_features": feature_names,
            },
        )

    frame = pd.DataFrame(records, columns=feature_names)
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise HTTPException(
            status_code=422,
            detail="All feature values must be finite numbers.",
        )

    return loaded.preprocessor.transform(frame).to_numpy(dtype=np.float32)


def _lstm_sequences_for_records(X: np.ndarray) -> np.ndarray:
    """Create one padded rolling sequence for every request record."""
    sequences = np.empty(
        (len(X), LSTM_TIMESTEPS, X.shape[1]),
        dtype=np.float32,
    )
    for index in range(len(X)):
        start = max(0, index - LSTM_TIMESTEPS + 1)
        history = X[start : index + 1]
        padding_size = LSTM_TIMESTEPS - len(history)
        if padding_size:
            padding = np.repeat(history[:1], padding_size, axis=0)
            history = np.vstack((padding, history))
        sequences[index] = history
    return sequences


def _predict_records(
    records: list[dict[str, float]],
    loaded: LoadedModel,
) -> list[PredictionResponse]:
    """Run inference for one or more request records."""
    X = _prepare_features(records, loaded)

    if loaded.model_type == "adaptive_lstm":
        sequences = _lstm_sequences_for_records(X)
        attack_probabilities = np.asarray(
            loaded.model.predict(sequences, verbose=0)
        ).reshape(-1)
        predictions = (attack_probabilities >= 0.5).astype(int)
        confidences = np.where(
            predictions == 1,
            attack_probabilities,
            1.0 - attack_probabilities,
        )
    else:
        predictions = np.asarray(loaded.model.predict(X)).astype(int)
        if hasattr(loaded.model, "predict_proba"):
            probabilities = np.asarray(loaded.model.predict_proba(X))
            class_to_column = {
                int(class_label): column
                for column, class_label in enumerate(loaded.model.classes_)
            }
            confidences = np.array(
                [
                    probabilities[row, class_to_column[int(prediction)]]
                    for row, prediction in enumerate(predictions)
                ]
            )
        else:
            confidences = np.full(len(predictions), np.nan)

    responses: list[PredictionResponse] = []
    for prediction, confidence in zip(predictions, confidences):
        responses.append(
            PredictionResponse(
                prediction=int(prediction),
                label="attack" if int(prediction) == 1 else "normal",
                model_name=loaded.model_name,
                model_version=loaded.version,
                confidence=(
                    float(confidence) if np.isfinite(confidence) else None
                ),
            )
        )
    return responses


def _artifact_metadata(path: Path) -> tuple[str, int | None] | None:
    """Classify model artifacts and ignore support files."""
    match = ADAPTIVE_LSTM_PATTERN.match(path.name)
    if match:
        return "adaptive_lstm", int(match.group(1))

    match = ADAPTIVE_RF_PATTERN.match(path.name)
    if match:
        return "adaptive_random_forest", int(match.group(1))

    if path.name == STATIC_RF_FILENAME:
        return "static_random_forest", None
    if path.name == "lstm_initial.keras":
        return "initial_lstm", 0
    return None


@app.get("/")
def project_info() -> dict[str, Any]:
    """Return project and endpoint information."""
    return {
        "project": "IoT Drift Online Learning",
        "task": "Binary IoT traffic classification with drift adaptation",
        "labels": {"0": "normal", "1": "attack"},
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return API and active-model health information."""
    loaded = state.loaded
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded is not None,
        "model_name": loaded.model_name if loaded else None,
        "model_version": loaded.version if loaded else None,
        "model_file": loaded.path.name if loaded else None,
        "error": state.load_error,
    }


@app.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    """List model and support artifacts in local cloud storage."""
    active_path = state.loaded.path.resolve() if state.loaded else None
    artifacts: list[ModelInfo] = []

    for path in sorted(CLOUD_MODEL_DIR.glob("*"), key=lambda item: item.name):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        metadata = _artifact_metadata(path)
        if metadata is None:
            continue
        model_type, version = metadata
        artifacts.append(
            ModelInfo(
                filename=path.name,
                model_type=model_type,
                version=version,
                size_bytes=path.stat().st_size,
                is_active=active_path == path.resolve(),
            )
        )
    return artifacts


@app.post("/predict", response_model=PredictionResponse)
def predict(request: FeatureRecord) -> PredictionResponse:
    """Predict one IoT feature record."""
    loaded = _require_loaded_model()
    return _predict_records([request.root], loaded)[0]


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(request: FeatureBatch) -> BatchPredictionResponse:
    """Predict a list of IoT records in their supplied stream order."""
    loaded = _require_loaded_model()
    return BatchPredictionResponse(
        predictions=_predict_records(request.root, loaded)
    )
