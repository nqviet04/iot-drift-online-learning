"""Evaluation metrics for static and adaptive experiments."""

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
