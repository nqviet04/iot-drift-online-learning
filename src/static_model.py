"""Static baseline model training utilities."""

from sklearn.ensemble import RandomForestClassifier


def build_static_model(random_state: int = 42) -> RandomForestClassifier:
    """Create the default static baseline classifier."""
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
