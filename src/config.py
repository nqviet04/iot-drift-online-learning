"""Central configuration for paths and experiment constants.

All filesystem paths are derived from the project root so the project can run
on different machines without hard-coded absolute paths.
"""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SYNTHETIC_DIR = DATA_DIR / "synthetic"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
LOG_DIR = OUTPUT_DIR / "logs"

CLOUD_MODEL_DIR = ROOT_DIR / "cloud_model_storage"
TON_IOT_RAW_DIR = DATA_RAW_DIR / "TON_IoT"
CIC_IOT_RAW_DIR = DATA_RAW_DIR / "CICIoT"

# Reproducibility and stream simulation settings.
RANDOM_STATE = 42
TRAIN_RATIO = 0.6
TEST_SIZE_BY_TIME = 0.4
WINDOW_SIZE = 1000
RECENT_BUFFER_SIZE = 5000
LSTM_TIMESTEPS = 10
BINARY_LABEL_COLUMN = "label_binary"

# AWS configuration is read from environment variables only.
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")

# Backward-compatible aliases for early skeleton modules.
PROJECT_ROOT = ROOT_DIR
RAW_DATA_DIR = DATA_RAW_DIR
PROCESSED_DATA_DIR = DATA_PROCESSED_DIR
SYNTHETIC_DATA_DIR = DATA_SYNTHETIC_DIR
FIGURES_DIR = FIGURE_DIR
METRICS_DIR = METRIC_DIR
LOGS_DIR = LOG_DIR
MODEL_STORAGE_DIR = CLOUD_MODEL_DIR
TON_IOT_DIR = TON_IOT_RAW_DIR
CICIOT_DIR = CIC_IOT_RAW_DIR
TARGET_COLUMN = BINARY_LABEL_COLUMN


def _ensure_project_dirs() -> None:
    """Create required local directories when this module is imported."""
    for directory in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_SYNTHETIC_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        METRIC_DIR,
        LOG_DIR,
        CLOUD_MODEL_DIR,
        TON_IOT_RAW_DIR,
        CIC_IOT_RAW_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


_ensure_project_dirs()
