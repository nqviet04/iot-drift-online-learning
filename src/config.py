"""Project configuration and path helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TON_IOT_DIR = RAW_DATA_DIR / "TON_IoT"
CICIOT_DIR = RAW_DATA_DIR / "CICIoT"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOGS_DIR = OUTPUT_DIR / "logs"

MODEL_STORAGE_DIR = PROJECT_ROOT / "cloud_model_storage"

RANDOM_STATE = 42
TARGET_COLUMN = "label"
