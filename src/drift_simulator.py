"""Helpers for building stream windows and synthetic concept drift scenarios."""

import pandas as pd


def split_by_time(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Sort a dataset by timestamp to emulate an IoT stream."""
    return df.sort_values(timestamp_column).reset_index(drop=True)
