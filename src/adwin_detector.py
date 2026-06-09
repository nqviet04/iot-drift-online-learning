"""ADWIN wrapper and detection-delay evaluation utilities."""

import pandas as pd

try:
    from river.drift import ADWIN
except ImportError as exc:
    raise ImportError(
        "river is required for ADWIN drift detection. "
        "Install project dependencies with: pip install -r requirements.txt"
    ) from exc


class ADWINDriftDetector:
    """Detect changes in a binary prediction-error stream using ADWIN."""

    def __init__(self, delta: float = 0.002) -> None:
        if not 0 < delta < 1:
            raise ValueError("delta must be between 0 and 1.")

        self.delta = delta
        self.detector = ADWIN(delta=delta)
        self.detected_drifts: list[int] = []

    def update(self, error_value: int, index: int) -> bool:
        """Update ADWIN with one prediction error and report a new drift."""
        if error_value not in (0, 1):
            raise ValueError("error_value must be 0 for correct or 1 for incorrect prediction.")

        self.detector.update(error_value)
        if self.detector.drift_detected:
            drift_index = int(index)
            self.detected_drifts.append(drift_index)
            return True
        return False

    def get_detected_drifts(self) -> list[int]:
        """Return a copy of all detected drift indices."""
        return self.detected_drifts.copy()

    def reset(self) -> None:
        """Reset ADWIN state and clear recorded drift indices."""
        self.detector = ADWIN(delta=self.delta)
        self.detected_drifts.clear()


def calculate_detection_delay(
    actual_drift_points: list[int],
    detected_drift_points: list[int],
) -> pd.DataFrame:
    """Match each actual drift with the first detection at or after it."""
    sorted_detected = sorted(int(point) for point in detected_drift_points)
    rows: list[dict[str, int | None]] = []

    for actual_point in actual_drift_points:
        actual = int(actual_point)
        detected = next((point for point in sorted_detected if point >= actual), None)
        rows.append(
            {
                "actual_drift_point": actual,
                "detected_drift_point": detected,
                "delay": detected - actual if detected is not None else None,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["actual_drift_point", "detected_drift_point", "delay"],
    )


def build_adwin(delta: float = 0.002) -> ADWIN:
    """Create a raw ADWIN instance for backward compatibility."""
    return ADWIN(delta=delta)
