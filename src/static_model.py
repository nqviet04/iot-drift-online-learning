"""Static baseline model training and persistence utilities."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.config import RANDOM_STATE


def train_random_forest(
    X_train,
    y_train,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Train a balanced Random Forest static baseline classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_sgd_classifier(
    X_train,
    y_train,
    random_state: int = RANDOM_STATE,
) -> SGDClassifier:
    """Train an SGD logistic classifier suitable for future partial_fit usage."""
    model = SGDClassifier(
        loss="log_loss",
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str | Path) -> Path:
    """Save a trained model to disk with joblib."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: str | Path):
    """Load a trained model from disk with joblib."""
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    return joblib.load(input_path)


def build_static_model(random_state: int = 42) -> RandomForestClassifier:
    """Create an unfitted Random Forest baseline for backward compatibility."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
