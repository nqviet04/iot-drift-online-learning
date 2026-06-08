"""Preprocessing utilities for binary IoT intrusion detection datasets."""

import pandas as pd


def normalize_binary_labels(df: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Return a copy with labels normalized to 0 for benign and 1 for attack."""
    output = df.copy()
    output[label_column] = output[label_column].map(
        lambda value: 0 if str(value).strip().lower() in {"0", "benign", "normal"} else 1
    )
    return output
