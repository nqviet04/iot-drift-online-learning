"""Helpers for stream ordering and synthetic concept drift scenarios."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BINARY_LABEL_COLUMN,
    DATA_SYNTHETIC_DIR,
    METRIC_DIR,
    RANDOM_STATE,
)


SYNTHETIC_FILENAME = "synthetic_iot_drift.csv"
DRIFT_POINTS_FILENAME = "synthetic_drift_points.json"


def split_by_time(df: pd.DataFrame, timestamp_column: str) -> pd.DataFrame:
    """Sort a dataset by timestamp to emulate an IoT stream."""
    return df.sort_values(timestamp_column).reset_index(drop=True)


def _stage_sizes(n_samples: int, n_stages: int = 4) -> list[int]:
    """Split samples as evenly as possible across drift stages."""
    base_size = n_samples // n_stages
    sizes = [base_size] * n_stages
    for idx in range(n_samples % n_stages):
        sizes[idx] += 1
    return sizes


def _sample_attack_types(
    rng: np.random.Generator,
    labels: np.ndarray,
    attack_type_probs: dict[str, float],
) -> np.ndarray:
    """Assign attack types while keeping benign rows explicit."""
    attack_types = np.full(labels.shape[0], "Benign", dtype=object)
    attack_mask = labels == 1

    if attack_mask.any():
        types = list(attack_type_probs.keys())
        probs = np.array(list(attack_type_probs.values()), dtype=float)
        probs = probs / probs.sum()
        attack_types[attack_mask] = rng.choice(types, size=attack_mask.sum(), p=probs)

    return attack_types


def _generate_stage_features(
    rng: np.random.Generator,
    stage_idx: int,
    labels: np.ndarray,
    attack_types: np.ndarray,
    n_features: int,
) -> np.ndarray:
    """Generate numeric IoT-like features for one drift stage."""
    n_rows = labels.shape[0]
    feature_idx = np.arange(n_features)

    benign_mean = np.sin(feature_idx / 3.0) * 0.4 + stage_idx * 0.15
    benign_std = 0.8 + (feature_idx % 5) * 0.03
    features = rng.normal(benign_mean, benign_std, size=(n_rows, n_features))

    attack_profiles = {
        "DoS": np.linspace(1.2, 2.8, n_features),
        "DDoS": np.r_[np.full(n_features // 2, 3.2), np.full(n_features - n_features // 2, 1.4)],
        "Recon": np.cos(feature_idx / 2.0) * 2.0,
        "Mirai": np.where(feature_idx % 2 == 0, 3.5, -1.8),
    }

    for attack_type, shift in attack_profiles.items():
        mask = attack_types == attack_type
        if not mask.any():
            continue

        stage_shift = shift + stage_idx * 0.35
        attack_noise = 0.9 + stage_idx * 0.15
        features[mask] = rng.normal(stage_shift, attack_noise, size=(mask.sum(), n_features))

    if stage_idx == 2:
        # Stage 3 has a stronger distribution change and feature interaction.
        features[:, : n_features // 2] *= 1.6
        features[:, n_features // 2 :] -= 1.2

    if stage_idx == 3:
        # Stage 4 is intentionally more mixed and harder to separate.
        mixed_component = rng.normal(0.0, 1.4, size=(n_rows, n_features))
        features = 0.65 * features + 0.35 * mixed_component
        if n_features >= 4:
            features[:, 0] += np.tanh(features[:, 1] * features[:, 2])
            features[:, 3] = np.sin(features[:, 3]) + rng.normal(0.0, 0.25, size=n_rows)

    return features


def generate_synthetic_iot_data(
    n_samples: int = 50000,
    n_features: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, list[int]]:
    """Generate a synthetic IoT stream with four concept drift stages.

    Returns a DataFrame containing timestamp, numeric features, attack_type,
    and label_binary, plus the true drift points between stages.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")
    if n_features <= 0:
        raise ValueError("n_features must be greater than 0.")

    rng = np.random.default_rng(random_state)
    stage_sizes = _stage_sizes(n_samples)
    drift_points = np.cumsum(stage_sizes)[:-1].tolist()

    stage_configs = [
        {
            "attack_ratio": 0.08,
            "attack_type_probs": {"DoS": 0.85, "DDoS": 0.10, "Recon": 0.05},
        },
        {
            "attack_ratio": 0.32,
            "attack_type_probs": {"DDoS": 0.75, "DoS": 0.15, "Recon": 0.10},
        },
        {
            "attack_ratio": 0.48,
            "attack_type_probs": {"Recon": 0.45, "Mirai": 0.35, "DDoS": 0.15, "DoS": 0.05},
        },
        {
            "attack_ratio": 0.42,
            "attack_type_probs": {"DDoS": 0.30, "Recon": 0.25, "Mirai": 0.25, "DoS": 0.20},
        },
    ]

    frames: list[pd.DataFrame] = []
    cursor = 0
    timestamps = pd.date_range(start="2024-01-01", periods=n_samples, freq="s")

    for stage_idx, (stage_size, config) in enumerate(zip(stage_sizes, stage_configs)):
        labels = rng.binomial(1, config["attack_ratio"], size=stage_size)
        attack_types = _sample_attack_types(rng, labels, config["attack_type_probs"])
        features = _generate_stage_features(rng, stage_idx, labels, attack_types, n_features)

        feature_columns = [f"feature_{idx}" for idx in range(n_features)]
        stage_df = pd.DataFrame(features, columns=feature_columns)
        stage_df["timestamp"] = timestamps[cursor : cursor + stage_size]
        stage_df["attack_type"] = attack_types
        stage_df[BINARY_LABEL_COLUMN] = labels.astype(int)

        frames.append(stage_df)
        cursor += stage_size

    df = pd.concat(frames, ignore_index=True)
    ordered_columns = (
        ["timestamp"]
        + [f"feature_{idx}" for idx in range(n_features)]
        + ["attack_type", BINARY_LABEL_COLUMN]
    )

    return df[ordered_columns], drift_points


def save_synthetic_dataset(
    n_samples: int = 50000,
    n_features: int = 20,
    random_state: int = RANDOM_STATE,
    csv_path: str | Path | None = None,
    drift_points_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[int], Path, Path]:
    """Generate and save the synthetic IoT drift dataset and drift metadata."""
    df, drift_points = generate_synthetic_iot_data(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )

    csv_output_path = Path(csv_path) if csv_path else DATA_SYNTHETIC_DIR / SYNTHETIC_FILENAME
    drift_output_path = (
        Path(drift_points_path) if drift_points_path else METRIC_DIR / DRIFT_POINTS_FILENAME
    )

    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    drift_output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_output_path, index=False)
    drift_payload = {
        "n_samples": n_samples,
        "n_features": n_features,
        "drift_points": drift_points,
        "description": "True drift points are row indices where the stream enters a new stage.",
    }
    drift_output_path.write_text(json.dumps(drift_payload, indent=2), encoding="utf-8")

    return df, drift_points, csv_output_path, drift_output_path
