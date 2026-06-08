"""Generate synthetic data for testing the pipeline before real datasets exist."""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import BINARY_LABEL_COLUMN
from src.drift_simulator import save_synthetic_dataset


def main() -> None:
    """Create synthetic IoT drift data and print a compact summary."""
    df, drift_points, csv_path, drift_points_path = save_synthetic_dataset()
    csv_display_path = csv_path.relative_to(ROOT_DIR)
    drift_display_path = drift_points_path.relative_to(ROOT_DIR)

    print("Synthetic IoT drift dataset generated successfully.")
    print(f"CSV path: {csv_display_path}")
    print(f"Drift points path: {drift_display_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nLabel distribution:")
    print(df[BINARY_LABEL_COLUMN].value_counts().sort_index().to_string())
    print("\nAttack type distribution:")
    print(df["attack_type"].value_counts().to_string())
    print(f"\nActual drift points: {drift_points}")


if __name__ == "__main__":
    main()
