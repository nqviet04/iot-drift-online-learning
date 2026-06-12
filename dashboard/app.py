"""Streamlit dashboard for IoT drift-detection experiment results.

Run from the project root:
    streamlit run dashboard/app.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    CLOUD_MODEL_DIR,
    DATA_SYNTHETIC_DIR,
    FIGURE_DIR,
    LOG_DIR,
    METRIC_DIR,
    OUTPUT_DIR,
)


SYNTHETIC_DATA_PATH = DATA_SYNTHETIC_DIR / "synthetic_iot_drift.csv"
STATIC_WINDOW_PATH = METRIC_DIR / "static_window_metrics.csv"
ADWIN_DRIFTS_PATH = METRIC_DIR / "adwin_detected_drifts.json"
ADWIN_DELAY_PATH = METRIC_DIR / "adwin_detection_delay.csv"
ADAPTIVE_STATIC_WINDOW_PATH = METRIC_DIR / "adaptive_static_window_metrics.csv"
ADAPTIVE_STATIC_RETRAIN_PATH = METRIC_DIR / "adaptive_static_retrain_log.csv"
ADAPTIVE_LSTM_WINDOW_PATH = METRIC_DIR / "adaptive_lstm_window_metrics.csv"
ADAPTIVE_LSTM_RETRAIN_PATH = METRIC_DIR / "adaptive_lstm_retrain_log.csv"
MODEL_COMPARISON_PATH = METRIC_DIR / "model_comparison_summary.csv"

SCRIPT_COMMANDS = {
    "generate_synthetic": {
        "label": "Generate Synthetic Dataset",
        "script": "scripts/00_generate_synthetic_data.py",
    },
    "train_static": {
        "label": "Train Static Model",
        "script": "scripts/02_train_static.py",
    },
    "run_adwin": {
        "label": "Run ADWIN Drift Detection",
        "script": "scripts/03_run_stream_adwin.py",
    },
    "adaptive_rf": {
        "label": "Run Adaptive Random Forest",
        "script": "scripts/04_run_adaptive_static.py",
    },
    "train_lstm": {
        "label": "Train Initial LSTM",
        "script": "scripts/05_train_lstm.py",
    },
    "adaptive_lstm": {
        "label": "Run Adaptive LSTM",
        "script": "scripts/06_run_adaptive_lstm.py",
    },
    "compare_models": {
        "label": "Compare Models",
        "script": "scripts/07_compare_models.py",
    },
    "generate_report": {
        "label": "Generate Experiment Report",
        "script": "scripts/08_generate_experiment_report.py",
    },
    "upload_azure": {
        "label": "Upload Models to Azure",
        "script": "scripts/09_upload_models_to_azure.py",
    },
}

PIPELINE_ARTIFACTS = {
    "generate_synthetic": (
        "Synthetic dataset",
        ROOT_DIR / "data" / "synthetic" / "synthetic_iot_drift.csv",
    ),
    "train_static": (
        "Static model",
        CLOUD_MODEL_DIR / "static_random_forest.joblib",
    ),
    "run_adwin": (
        "ADWIN detection delay",
        METRIC_DIR / "adwin_detection_delay.csv",
    ),
    "adaptive_rf": (
        "Adaptive RF summary",
        METRIC_DIR / "adaptive_static_summary.json",
    ),
    "train_lstm": (
        "Initial LSTM model",
        CLOUD_MODEL_DIR / "lstm_initial.keras",
    ),
    "adaptive_lstm": (
        "Adaptive LSTM summary",
        METRIC_DIR / "adaptive_lstm_summary.json",
    ),
    "compare_models": (
        "Model comparison",
        METRIC_DIR / "model_comparison_summary.csv",
    ),
    "generate_report": (
        "Experiment report",
        ROOT_DIR / "outputs" / "experiment_report.md",
    ),
}

BASE_FULL_PIPELINE = [
    "generate_synthetic",
    "train_static",
    "run_adwin",
    "adaptive_rf",
]
LSTM_PIPELINE = ["train_lstm", "adaptive_lstm"]
FINAL_PIPELINE = ["compare_models", "generate_report"]

DEMO_BACKUP_DIR = ROOT_DIR / "demo_backups"
DEMO_BACKUP_SOURCES = (
    OUTPUT_DIR,
    CLOUD_MODEL_DIR,
    DATA_SYNTHETIC_DIR,
)
RESET_TARGETS = {
    "outputs": {
        "label": "outputs/",
        "path": OUTPUT_DIR,
        "mode": "contents",
    },
    "models": {
        "label": "cloud_model_storage/",
        "path": CLOUD_MODEL_DIR,
        "mode": "contents",
    },
    "synthetic": {
        "label": "Synthetic CSV/JSON files",
        "path": DATA_SYNTHETIC_DIR,
        "mode": "synthetic_files",
    },
}
RECREATED_DEMO_DIRS = (
    METRIC_DIR,
    FIGURE_DIR,
    LOG_DIR,
    CLOUD_MODEL_DIR,
    DATA_SYNTHETIC_DIR,
)
SYNTHETIC_RESET_SUFFIXES = {".csv", ".json"}

METRIC_OPTIONS = {
    "F1": "f1",
    "Accuracy": "accuracy",
    "Recall": "recall",
}


st.set_page_config(
    page_title="IoT Drift Online Learning",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, modified_time: float) -> pd.DataFrame:
    """Read a CSV and invalidate cache when the file changes."""
    del modified_time
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json_cached(path: str, modified_time: float) -> Any:
    """Read JSON and invalidate cache when the file changes."""
    del modified_time
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path: Path, label: str) -> pd.DataFrame | None:
    """Load a CSV or show a friendly dashboard warning."""
    if not path.exists():
        st.warning(f"Chưa có {label}: `{path.relative_to(ROOT_DIR)}`")
        return None
    try:
        return _read_csv_cached(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Không thể đọc {label}: {exc}")
        return None


def load_json(path: Path, label: str) -> Any | None:
    """Load JSON or show a friendly dashboard warning."""
    if not path.exists():
        st.warning(f"Chưa có {label}: `{path.relative_to(ROOT_DIR)}`")
        return None
    try:
        return _read_json_cached(str(path), path.stat().st_mtime)
    except Exception as exc:
        st.warning(f"Không thể đọc {label}: {exc}")
        return None


def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    label: str,
) -> bool:
    """Validate plotting columns without crashing the dashboard."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        st.warning(f"{label} thiếu cột: {missing}")
        return False
    return True


def line_figure(
    df: pd.DataFrame,
    series: dict[str, str],
    title: str,
    y_title: str,
) -> go.Figure:
    """Build a consistent multi-series stream-window line chart."""
    fig = go.Figure()
    for display_name, column in series.items():
        if column not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["window_id"],
                y=df[column],
                mode="lines+markers",
                name=display_name,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Window ID",
        yaxis_title=y_title,
        yaxis_range=[0, 1.05],
        legend_title_text="Model",
        margin=dict(l=20, r=20, t=55, b=20),
        height=430,
    )
    return fig


def add_retrain_markers(
    fig: go.Figure,
    window_df: pd.DataFrame,
    retrain_df: pd.DataFrame | None,
) -> None:
    """Add vertical markers for retraining events."""
    if retrain_df is None or retrain_df.empty:
        return
    if not require_columns(
        retrain_df,
        ["detected_drift_index"],
        "Retrain log",
    ):
        return

    for row in retrain_df.itertuples(index=False):
        drift_index = int(row.detected_drift_index)
        matching_window = window_df[
            (window_df["start_index"] <= drift_index)
            & (window_df["end_index"] >= drift_index)
        ]
        if matching_window.empty:
            continue
        window_id = float(matching_window.iloc[0]["window_id"])
        version = getattr(row, "version", None)
        fig.add_vline(
            x=window_id,
            line_dash="dot",
            line_color="#7c3aed",
            annotation_text=f"Retrain v{version}" if version is not None else "Retrain",
            annotation_position="top",
        )


def metric_file_options() -> list[Path]:
    """Return all readable metrics files for the sidebar viewer."""
    if not METRIC_DIR.exists():
        return []
    return sorted(
        [
            path
            for path in METRIC_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".json"}
        ],
        key=lambda path: path.name,
    )


def model_inventory() -> pd.DataFrame:
    """Build a compact local model-storage inventory."""
    rows = []
    if CLOUD_MODEL_DIR.exists():
        for path in sorted(CLOUD_MODEL_DIR.iterdir(), key=lambda item: item.name):
            if not path.is_file() or path.name == ".gitkeep":
                continue
            rows.append(
                {
                    "model": path.name,
                    "size_kb": round(path.stat().st_size / 1024, 1),
                }
            )
    return pd.DataFrame(rows)


def _format_time(value: datetime) -> str:
    """Format one local timestamp for pipeline logs."""
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _redact_sensitive_text(text: str | None) -> str:
    """Remove cloud secrets from text before rendering it in the dashboard."""
    if not text:
        return ""

    redacted = str(text)
    for variable_name in (
        "AZURE_STORAGE_CONNECTION_STRING",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_ACCESS_KEY_ID",
    ):
        value = os.getenv(variable_name)
        if value:
            redacted = redacted.replace(value, f"[REDACTED {variable_name}]")

    redacted = re.sub(
        r"(?i)(AccountKey=)[^;\s]+",
        r"\1[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(SharedAccessSignature=)[^;\s]+",
        r"\1[REDACTED]",
        redacted,
    )
    return redacted


def _artifact_status(step_key: str) -> dict[str, Any] | None:
    """Return status metadata for the important output of one pipeline step."""
    artifact = PIPELINE_ARTIFACTS.get(step_key)
    if artifact is None:
        return None

    label, path = artifact
    exists = path.exists()
    return {
        "label": label,
        "path": str(path.relative_to(ROOT_DIR)),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def pipeline_status_frame() -> pd.DataFrame:
    """Build a table showing which expected pipeline artifacts exist."""
    rows = []
    for step_key, (label, path) in PIPELINE_ARTIFACTS.items():
        exists = path.exists()
        rows.append(
            {
                "Stage": SCRIPT_COMMANDS[step_key]["label"],
                "Artifact": label,
                "Path": str(path.relative_to(ROOT_DIR)),
                "Status": "Available" if exists else "Missing",
                "Size (KB)": (
                    round(path.stat().st_size / 1024, 1) if exists else None
                ),
            }
        )
    return pd.DataFrame(rows)


def _project_relative(path: Path) -> str:
    """Return a stable project-relative display path."""
    return str(path.resolve().relative_to(ROOT_DIR.resolve()))


def _assert_safe_reset_root(path: Path) -> Path:
    """Validate that a reset root is one of the fixed allow-listed paths."""
    resolved = path.resolve()
    allowed = {
        Path(specification["path"]).resolve()
        for specification in RESET_TARGETS.values()
    }
    if resolved not in allowed:
        raise ValueError("Reset target is outside the fixed allowlist.")
    if resolved == ROOT_DIR.resolve():
        raise ValueError("Project root cannot be used as a reset target.")
    return resolved


def _next_backup_directory() -> Path:
    """Create a timestamped backup directory without overwriting an old run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = DEMO_BACKUP_DIR / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = DEMO_BACKUP_DIR / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def backup_current_results() -> dict[str, Any]:
    """Copy generated demo artifacts to a timestamped local backup."""
    result: dict[str, Any] = {
        "backup_dir": None,
        "backed_up": [],
        "skipped": [],
        "status": "Success",
        "error": None,
    }

    try:
        backup_dir = _next_backup_directory()
        result["backup_dir"] = _project_relative(backup_dir)

        for source in DEMO_BACKUP_SOURCES:
            if not source.exists():
                result["skipped"].append(
                    f"{_project_relative(source)} (not found)"
                )
                continue

            destination = backup_dir / source.relative_to(ROOT_DIR)
            if source.is_dir():
                shutil.copytree(
                    source,
                    destination,
                    dirs_exist_ok=True,
                    symlinks=True,
                )
                copied_files = [
                    path
                    for path in source.rglob("*")
                    if path.is_file() and path.name != ".gitkeep"
                ]
                if copied_files:
                    result["backed_up"].extend(
                        _project_relative(path) for path in copied_files
                    )
                else:
                    result["skipped"].append(
                        f"{_project_relative(source)} (no generated files)"
                    )
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                result["backed_up"].append(_project_relative(source))
    except Exception as exc:
        result["status"] = "Failed"
        result["error"] = (
            "Backup failed safely. "
            f"Error type: {type(exc).__name__}."
        )
    return result


def _delete_allowlisted_contents(
    root: Path,
    deleted: list[str],
    skipped: list[str],
) -> None:
    """Delete generated children of one allow-listed directory."""
    safe_root = _assert_safe_reset_root(root)
    safe_root.mkdir(parents=True, exist_ok=True)

    children = sorted(
        safe_root.rglob("*"),
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    )
    for child in children:
        display_path = _project_relative(child)
        if child.name == ".gitkeep":
            skipped.append(f"{display_path} (preserved)")
            continue

        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
                deleted.append(display_path)
            elif child.is_dir():
                try:
                    child.rmdir()
                    deleted.append(f"{display_path}/")
                except OSError:
                    skipped.append(
                        f"{display_path}/ (preserved because not empty)"
                    )
            else:
                skipped.append(f"{display_path} (unsupported file type)")
        except OSError as exc:
            skipped.append(
                f"{display_path} (could not delete: {type(exc).__name__})"
            )


def _delete_synthetic_files(
    root: Path,
    deleted: list[str],
    skipped: list[str],
) -> None:
    """Delete only CSV/JSON files from the synthetic-data allowlist."""
    safe_root = _assert_safe_reset_root(root)
    safe_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        safe_root.rglob("*"),
        key=lambda path: str(path),
        reverse=True,
    )
    for path in candidates:
        display_path = _project_relative(path)
        if path.is_symlink():
            skipped.append(f"{display_path} (symlink preserved)")
        elif path.is_file() and path.suffix.lower() in SYNTHETIC_RESET_SUFFIXES:
            try:
                path.unlink()
                deleted.append(display_path)
            except OSError as exc:
                skipped.append(
                    f"{display_path} "
                    f"(could not delete: {type(exc).__name__})"
                )
        elif path.is_file() and path.name != ".gitkeep":
            skipped.append(f"{display_path} (non CSV/JSON file preserved)")


def reset_demo_outputs(selected_targets: list[str]) -> dict[str, Any]:
    """Reset selected generated artifacts using fixed allow-listed roots."""
    result: dict[str, Any] = {
        "status": "Success",
        "deleted": [],
        "skipped": [],
        "recreated": [],
        "error": None,
    }

    unknown_targets = set(selected_targets) - set(RESET_TARGETS)
    if unknown_targets:
        result["status"] = "Failed"
        result["error"] = "Reset rejected an unknown target."
        return result

    try:
        for target_key, specification in RESET_TARGETS.items():
            if target_key not in selected_targets:
                result["skipped"].append(
                    f"{specification['label']} (not selected)"
                )
                continue

            target_path = Path(specification["path"])
            if specification["mode"] == "contents":
                _delete_allowlisted_contents(
                    target_path,
                    result["deleted"],
                    result["skipped"],
                )
            else:
                _delete_synthetic_files(
                    target_path,
                    result["deleted"],
                    result["skipped"],
                )

        for directory in RECREATED_DEMO_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            result["recreated"].append(_project_relative(directory))
    except Exception as exc:
        result["status"] = "Failed"
        result["error"] = (
            "Reset stopped safely. "
            f"Error type: {type(exc).__name__}."
        )

    st.cache_data.clear()
    return result


def run_fixed_script(step_key: str) -> dict[str, Any]:
    """Run one allow-listed project script and capture a safe result."""
    if step_key not in SCRIPT_COMMANDS:
        raise ValueError("Unknown pipeline step.")

    specification = SCRIPT_COMMANDS[step_key]
    script_path = ROOT_DIR / specification["script"]
    display_command = f"python {specification['script']}"
    started_at = datetime.now().astimezone()

    result: dict[str, Any] = {
        "step_key": step_key,
        "label": specification["label"],
        "command": display_command,
        "started_at": _format_time(started_at),
        "finished_at": None,
        "duration_seconds": None,
        "status": "Failed",
        "return_code": None,
        "stdout": "",
        "stderr": "",
        "artifact": None,
    }

    if not script_path.exists():
        finished_at = datetime.now().astimezone()
        result.update(
            {
                "finished_at": _format_time(finished_at),
                "duration_seconds": (
                    finished_at - started_at
                ).total_seconds(),
                "stderr": f"Script not found: {specification['script']}",
                "artifact": _artifact_status(step_key),
            }
        )
        return result

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            shell=False,
            timeout=7200,
        )
        result.update(
            {
                "status": (
                    "Success" if completed.returncode == 0 else "Failed"
                ),
                "return_code": completed.returncode,
                "stdout": _redact_sensitive_text(completed.stdout),
                "stderr": _redact_sensitive_text(completed.stderr),
            }
        )
    except subprocess.TimeoutExpired as exc:
        result["stderr"] = _redact_sensitive_text(
            f"Script exceeded the 2-hour safety timeout. {exc.stderr or ''}"
        )
    except Exception as exc:
        result["stderr"] = (
            "Không thể chạy script. "
            f"Error type: {type(exc).__name__}."
        )

    finished_at = datetime.now().astimezone()
    result.update(
        {
            "finished_at": _format_time(finished_at),
            "duration_seconds": round(
                (finished_at - started_at).total_seconds(),
                3,
            ),
            "artifact": _artifact_status(step_key),
        }
    )
    return result


def _store_pipeline_result(result: dict[str, Any]) -> None:
    """Keep recent pipeline logs across Streamlit reruns."""
    history = st.session_state.setdefault("admin_pipeline_history", [])
    history.insert(0, result)
    del history[20:]


def render_pipeline_result(result: dict[str, Any]) -> None:
    """Render one completed script result without raising UI exceptions."""
    if result["status"] == "Success":
        st.success(f"{result['label']}: Success")
    else:
        st.error(f"{result['label']}: Failed")

    st.code(result["command"], language="powershell")
    time_col1, time_col2, duration_col = st.columns(3)
    time_col1.metric("Started", result["started_at"])
    time_col2.metric("Finished", result["finished_at"] or "Unknown")
    duration_col.metric(
        "Duration",
        f"{float(result['duration_seconds'] or 0):.2f} s",
    )

    artifact = result.get("artifact")
    if artifact is not None:
        if artifact["exists"]:
            st.success(
                f"Output available: `{artifact['path']}` "
                f"({artifact['size_bytes'] / 1024:.1f} KB)"
            )
        else:
            st.warning(f"Expected output is missing: `{artifact['path']}`")

    with st.expander("stdout", expanded=result["status"] == "Failed"):
        st.code(result["stdout"] or "(empty)", language="text")

    if result["stderr"]:
        with st.expander("stderr", expanded=True):
            st.code(result["stderr"], language="text")


def run_pipeline_steps(step_keys: list[str]) -> list[dict[str, Any]]:
    """Run fixed pipeline steps sequentially and stop after the first failure."""
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0, text="Preparing pipeline...")

    for index, step_key in enumerate(step_keys):
        specification = SCRIPT_COMMANDS[step_key]
        with st.status(
            f"Running: {specification['label']}",
            expanded=True,
        ) as status:
            st.code(
                f"python {specification['script']}",
                language="powershell",
            )
            st.write(f"Started: {_format_time(datetime.now().astimezone())}")
            result = run_fixed_script(step_key)
            _store_pipeline_result(result)
            results.append(result)

            if result["status"] == "Success":
                status.update(
                    label=f"{specification['label']}: Success",
                    state="complete",
                )
            else:
                status.update(
                    label=f"{specification['label']}: Failed",
                    state="error",
                )

        render_pipeline_result(result)
        progress.progress(
            (index + 1) / len(step_keys),
            text=f"Completed {index + 1}/{len(step_keys)} step(s)",
        )
        if result["status"] != "Success":
            st.warning("Pipeline stopped after the failed step.")
            break

    st.cache_data.clear()
    return results


st.title("Phát hiện Drift trong hệ thống IoT bằng Online Learning")
st.caption(
    "So sánh static model và adaptive model, phát hiện drift bằng ADWIN, "
    "và fine-tune LSTM theo luồng dữ liệu."
)

metric_files = metric_file_options()
st.sidebar.header("Dashboard Controls")
selected_metric_label = st.sidebar.selectbox(
    "Biểu đồ",
    options=list(METRIC_OPTIONS),
)
selected_metric = METRIC_OPTIONS[selected_metric_label]

selected_file = st.sidebar.selectbox(
    "Metrics file",
    options=metric_files,
    format_func=lambda path: path.name,
    index=0 if metric_files else None,
    placeholder="Chưa có metrics file",
)

st.sidebar.subheader("Local Models")
models_df = model_inventory()
if models_df.empty:
    st.sidebar.warning("Chưa có model trong cloud_model_storage/.")
else:
    st.sidebar.dataframe(
        models_df,
        hide_index=True,
        width="stretch",
    )

if selected_file is not None:
    with st.expander(f"Metrics viewer: {selected_file.name}"):
        if selected_file.suffix.lower() == ".csv":
            selected_data = load_csv(selected_file, selected_file.name)
            if selected_data is not None:
                st.dataframe(selected_data, width="stretch")
        else:
            selected_data = load_json(selected_file, selected_file.name)
            if selected_data is not None:
                st.json(selected_data)

(
    dataset_tab,
    static_tab,
    adwin_tab,
    adaptive_tab,
    comparison_tab,
    admin_tab,
) = st.tabs(
    [
        "Dataset Overview",
        "Static Model Performance",
        "ADWIN Drift Detection",
        "Adaptive Model",
        "Model Comparison",
        "Admin Pipeline",
    ]
)


with dataset_tab:
    dataset_df = load_csv(SYNTHETIC_DATA_PATH, "synthetic dataset")
    if dataset_df is not None:
        feature_columns = [
            column for column in dataset_df.columns if column.startswith("feature_")
        ]
        normal_count = (
            int((dataset_df["label_binary"] == 0).sum())
            if "label_binary" in dataset_df.columns
            else 0
        )
        attack_count = (
            int((dataset_df["label_binary"] == 1).sum())
            if "label_binary" in dataset_df.columns
            else 0
        )

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Rows", f"{len(dataset_df):,}")
        metric_col2.metric("Numeric features", len(feature_columns))
        metric_col3.metric("Normal", f"{normal_count:,}")
        metric_col4.metric("Attack", f"{attack_count:,}")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if "label_binary" in dataset_df.columns:
                label_counts = (
                    dataset_df["label_binary"]
                    .map({0: "Normal", 1: "Attack"})
                    .value_counts()
                    .rename_axis("label")
                    .reset_index(name="count")
                )
                fig = px.bar(
                    label_counts,
                    x="label",
                    y="count",
                    color="label",
                    title="Normal vs Attack",
                    color_discrete_map={
                        "Normal": "#2563eb",
                        "Attack": "#dc2626",
                    },
                )
                fig.update_layout(showlegend=False, height=390)
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Synthetic dataset chưa có cột `label_binary`.")

        with chart_col2:
            if "attack_type" in dataset_df.columns:
                attack_counts = (
                    dataset_df["attack_type"]
                    .value_counts()
                    .rename_axis("attack_type")
                    .reset_index(name="count")
                )
                fig = px.bar(
                    attack_counts,
                    x="attack_type",
                    y="count",
                    color="attack_type",
                    title="Attack Type Distribution",
                )
                fig.update_layout(showlegend=False, height=390)
                st.plotly_chart(fig, width="stretch")
            else:
                st.warning("Synthetic dataset chưa có cột `attack_type`.")

        with st.expander("Dataset sample"):
            st.dataframe(dataset_df.head(100), width="stretch")


with static_tab:
    static_df = load_csv(STATIC_WINDOW_PATH, "static window metrics")
    if static_df is not None:
        st.dataframe(static_df, width="stretch", hide_index=True)
        if require_columns(
            static_df,
            ["window_id", selected_metric],
            "Static metrics",
        ):
            fig = line_figure(
                static_df,
                {f"Static Random Forest {selected_metric_label}": selected_metric},
                f"Static Model {selected_metric_label} by Window",
                selected_metric_label,
            )
            st.plotly_chart(fig, width="stretch")


with adwin_tab:
    drift_payload = load_json(ADWIN_DRIFTS_PATH, "ADWIN detected drifts")
    delay_df = load_csv(ADWIN_DELAY_PATH, "ADWIN detection delay")

    actual_drifts: list[int] = []
    detected_drifts: list[int] = []
    if isinstance(drift_payload, dict):
        actual_drifts = [
            int(value) for value in drift_payload.get("actual_drift_points", [])
        ]
        detected_drifts = [
            int(value) for value in drift_payload.get("detected_drifts", [])
        ]

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual drifts", len(actual_drifts))
        col2.metric("Detected drifts", len(detected_drifts))
        col3.metric(
            "Error rate",
            f"{float(drift_payload.get('overall_error_rate', 0.0)):.4f}",
        )

    if delay_df is not None:
        st.subheader("Detection Delay")
        st.dataframe(delay_df, width="stretch", hide_index=True)

    drift_rows = [
        {"drift_point": point, "type": "Actual drift"}
        for point in actual_drifts
    ] + [
        {"drift_point": point, "type": "Detected drift"}
        for point in detected_drifts
    ]
    if drift_rows:
        drift_df = pd.DataFrame(drift_rows)
        fig = px.scatter(
            drift_df,
            x="drift_point",
            y="type",
            color="type",
            symbol="type",
            title="Actual vs Detected Drift Points",
            color_discrete_map={
                "Actual drift": "#dc2626",
                "Detected drift": "#7c3aed",
            },
        )
        fig.update_traces(marker_size=14)
        fig.update_layout(height=350, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    elif drift_payload is not None:
        st.info("ADWIN chưa phát hiện drift trên error stream hiện tại.")


with adaptive_tab:
    adaptive_source = st.radio(
        "Adaptive experiment",
        options=["Adaptive LSTM", "Adaptive Random Forest"],
        horizontal=True,
    )

    if adaptive_source == "Adaptive LSTM":
        adaptive_window_path = ADAPTIVE_LSTM_WINDOW_PATH
        adaptive_retrain_path = ADAPTIVE_LSTM_RETRAIN_PATH
    else:
        adaptive_window_path = ADAPTIVE_STATIC_WINDOW_PATH
        adaptive_retrain_path = ADAPTIVE_STATIC_RETRAIN_PATH

    adaptive_df = load_csv(
        adaptive_window_path,
        f"{adaptive_source} window metrics",
    )
    retrain_df = load_csv(
        adaptive_retrain_path,
        f"{adaptive_source} retrain log",
    )

    if adaptive_df is not None:
        static_column = f"static_{selected_metric}"
        adaptive_column = f"adaptive_{selected_metric}"
        st.dataframe(adaptive_df, width="stretch", hide_index=True)

        required = [
            "window_id",
            "start_index",
            "end_index",
            static_column,
            adaptive_column,
        ]
        if require_columns(adaptive_df, required, adaptive_source):
            fig = line_figure(
                adaptive_df,
                {
                    "Static": static_column,
                    "Adaptive": adaptive_column,
                },
                f"{adaptive_source}: {selected_metric_label} by Window",
                selected_metric_label,
            )
            add_retrain_markers(fig, adaptive_df, retrain_df)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Retraining Log")
    if retrain_df is None:
        pass
    elif retrain_df.empty:
        st.info("Experiment này chưa có lần retrain nào.")
    else:
        st.dataframe(retrain_df, width="stretch", hide_index=True)


with comparison_tab:
    comparison_df = load_csv(
        MODEL_COMPARISON_PATH,
        "model comparison summary",
    )
    if comparison_df is not None:
        st.dataframe(comparison_df, width="stretch", hide_index=True)
        average_metric = f"average_{selected_metric}"

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            if require_columns(
                comparison_df,
                ["model_name", average_metric],
                "Model comparison",
            ):
                fig = px.bar(
                    comparison_df,
                    x="model_name",
                    y=average_metric,
                    color="model_name",
                    title=f"Average {selected_metric_label}",
                    text_auto=".4f",
                )
                fig.update_layout(
                    showlegend=False,
                    yaxis_range=[0, 1.05],
                    height=420,
                )
                st.plotly_chart(fig, width="stretch")

        with chart_col2:
            if require_columns(
                comparison_df,
                [
                    "model_name",
                    average_metric,
                    "total_retrain_time_seconds",
                ],
                "Update-cost comparison",
            ):
                fig = px.scatter(
                    comparison_df,
                    x="total_retrain_time_seconds",
                    y=average_metric,
                    color="model_name",
                    symbol="model_name",
                    text="model_name",
                    title=f"{selected_metric_label} vs Update Cost",
                    size_max=18,
                )
                fig.update_traces(marker_size=13, textposition="top center")
                fig.update_layout(
                    xaxis_title="Total retrain time (seconds)",
                    yaxis_title=f"Average {selected_metric_label}",
                    yaxis_range=[0, 1.05],
                    height=420,
                    showlegend=False,
                )
                st.plotly_chart(fig, width="stretch")

    static_compare = load_csv(STATIC_WINDOW_PATH, "static metrics for comparison")
    adaptive_static_compare = load_csv(
        ADAPTIVE_STATIC_WINDOW_PATH,
        "adaptive RF metrics for comparison",
    )
    adaptive_lstm_compare = load_csv(
        ADAPTIVE_LSTM_WINDOW_PATH,
        "adaptive LSTM metrics for comparison",
    )

    comparison_fig = go.Figure()
    if (
        static_compare is not None
        and selected_metric in static_compare.columns
    ):
        comparison_fig.add_trace(
            go.Scatter(
                x=static_compare["window_id"],
                y=static_compare[selected_metric],
                mode="lines+markers",
                name="Static Random Forest",
            )
        )
    for model_name, frame in [
        ("Adaptive Random Forest", adaptive_static_compare),
        ("Adaptive LSTM", adaptive_lstm_compare),
    ]:
        column = f"adaptive_{selected_metric}"
        if frame is not None and column in frame.columns:
            comparison_fig.add_trace(
                go.Scatter(
                    x=frame["window_id"],
                    y=frame[column],
                    mode="lines+markers",
                    name=model_name,
                )
            )

    if comparison_fig.data:
        comparison_fig.update_layout(
            title=f"{selected_metric_label} over Stream Windows",
            xaxis_title="Window ID",
            yaxis_title=selected_metric_label,
            yaxis_range=[0, 1.05],
            height=450,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(comparison_fig, width="stretch")
    else:
        st.warning("Chưa đủ window metrics để vẽ model comparison.")


with admin_tab:
    st.subheader("Admin Pipeline")
    st.warning(
        "LSTM training và adaptive fine-tuning có thể mất vài phút. "
        "Không đóng dashboard khi pipeline đang chạy."
    )
    st.error(
        "Không public Admin Pipeline nếu dashboard được deploy lên internet. "
        "Tab này có quyền chạy các script huấn luyện trên máy chủ."
    )
    st.caption(
        "Chỉ các script cố định của project được phép chạy. "
        "Dashboard không nhận command tùy ý."
    )

    button_columns = st.columns(2)
    clicked_step: str | None = None
    individual_steps = list(SCRIPT_COMMANDS)
    for index, step_key in enumerate(individual_steps):
        specification = SCRIPT_COMMANDS[step_key]
        with button_columns[index % 2]:
            if st.button(
                specification["label"],
                key=f"admin_run_{step_key}",
                width="stretch",
            ):
                clicked_step = step_key

    st.divider()
    st.subheader("Run Full Pipeline")
    option_col1, option_col2 = st.columns(2)
    include_lstm = option_col1.checkbox(
        "Include LSTM",
        value=False,
        help="Run initial LSTM training and adaptive LSTM fine-tuning.",
    )
    upload_after_completion = option_col2.checkbox(
        "Upload to Azure after completion",
        value=False,
        help="Run scripts/09_upload_models_to_azure.py after local steps.",
    )
    run_full_pipeline = st.button(
        "Run Full Pipeline",
        key="admin_run_full_pipeline",
        type="primary",
        width="stretch",
    )

    if clicked_step is not None:
        run_pipeline_steps([clicked_step])

    if run_full_pipeline:
        full_pipeline_steps = BASE_FULL_PIPELINE.copy()
        if include_lstm:
            full_pipeline_steps.extend(LSTM_PIPELINE)
        full_pipeline_steps.extend(FINAL_PIPELINE)
        if upload_after_completion:
            full_pipeline_steps.append("upload_azure")
        run_pipeline_steps(full_pipeline_steps)

    st.divider()
    st.subheader("Demo Reset Tools")
    st.caption(
        "Backup và reset chỉ áp dụng cho artifact demo local. "
        "Dữ liệu trong Azure Blob Storage không bị xóa."
    )

    if st.button(
        "Backup Current Results",
        key="admin_backup_current_results",
        width="stretch",
    ):
        with st.spinner("Backing up current demo results..."):
            backup_result = backup_current_results()
        st.session_state["demo_backup_result"] = backup_result

    backup_result = st.session_state.get("demo_backup_result")
    if backup_result:
        if backup_result["status"] == "Success":
            st.success(
                "Backup completed: "
                f"`{backup_result['backup_dir']}`"
            )
        else:
            st.error(backup_result["error"])

        with st.expander(
            f"Backed up ({len(backup_result['backed_up'])})",
            expanded=False,
        ):
            if backup_result["backed_up"]:
                st.code(
                    "\n".join(backup_result["backed_up"]),
                    language="text",
                )
            else:
                st.write("No generated files were available to back up.")

        if backup_result["skipped"]:
            with st.expander(
                f"Backup skipped ({len(backup_result['skipped'])})"
            ):
                st.code(
                    "\n".join(backup_result["skipped"]),
                    language="text",
                )

    st.warning(
        "Reset Demo Outputs permanently deletes selected local generated files. "
        "Use Backup Current Results first when old experiment results matter."
    )

    reset_option_columns = st.columns(3)
    reset_outputs = reset_option_columns[0].checkbox(
        "Delete outputs/",
        value=True,
        key="admin_reset_outputs",
    )
    reset_models = reset_option_columns[1].checkbox(
        "Delete cloud_model_storage/",
        value=True,
        key="admin_reset_models",
    )
    reset_synthetic = reset_option_columns[2].checkbox(
        "Delete synthetic CSV/JSON",
        value=True,
        key="admin_reset_synthetic",
    )
    reset_acknowledged = st.checkbox(
        "I understand this will delete generated demo files",
        value=False,
        key="admin_reset_acknowledged",
    )
    reset_confirmation = st.text_input(
        "Type RESET to confirm",
        value="",
        key="admin_reset_confirmation",
        autocomplete="off",
    )
    selected_reset_targets = [
        target
        for target, selected in (
            ("outputs", reset_outputs),
            ("models", reset_models),
            ("synthetic", reset_synthetic),
        )
        if selected
    ]
    reset_ready = (
        bool(selected_reset_targets)
        and reset_acknowledged
        and reset_confirmation == "RESET"
    )
    reset_clicked = st.button(
        "Reset Demo Outputs",
        key="admin_reset_demo_outputs",
        type="primary",
        disabled=not reset_ready,
        width="stretch",
    )

    if reset_clicked:
        with st.spinner("Resetting selected local demo artifacts..."):
            reset_result = reset_demo_outputs(selected_reset_targets)
        st.session_state["demo_reset_result"] = reset_result
        st.session_state["demo_reset_completed"] = (
            reset_result["status"] == "Success"
        )

    reset_result = st.session_state.get("demo_reset_result")
    if reset_result:
        if reset_result["status"] == "Success":
            st.success("Local demo reset completed.")
        else:
            st.error(reset_result["error"])

        reset_result_columns = st.columns(3)
        reset_result_columns[0].metric(
            "Deleted",
            len(reset_result["deleted"]),
        )
        reset_result_columns[1].metric(
            "Skipped",
            len(reset_result["skipped"]),
        )
        reset_result_columns[2].metric(
            "Recreated",
            len(reset_result["recreated"]),
        )

        for title, key in (
            ("Deleted files/folders", "deleted"),
            ("Skipped files/folders", "skipped"),
            ("Recreated folders", "recreated"),
        ):
            with st.expander(
                f"{title} ({len(reset_result[key])})",
                expanded=key == "deleted",
            ):
                st.code(
                    "\n".join(reset_result[key]) or "(none)",
                    language="text",
                )

    st.subheader("Run From Scratch")
    scratch_include_lstm = st.checkbox(
        "Include LSTM",
        value=False,
        key="admin_scratch_include_lstm",
    )
    scratch_ready = bool(
        st.session_state.get("demo_reset_completed", False)
    )
    if not scratch_ready:
        st.info(
            "Reset Demo Outputs successfully in this session to enable "
            "Run Full Pipeline From Scratch."
        )
    run_from_scratch = st.button(
        "Run Full Pipeline From Scratch",
        key="admin_run_from_scratch",
        disabled=not scratch_ready,
        width="stretch",
    )
    if run_from_scratch:
        scratch_steps = BASE_FULL_PIPELINE.copy()
        if scratch_include_lstm:
            scratch_steps.extend(LSTM_PIPELINE)
        scratch_steps.extend(FINAL_PIPELINE)
        run_pipeline_steps(scratch_steps)

    st.divider()
    st.subheader("Pipeline Status")
    status_df = pipeline_status_frame()
    available_count = int((status_df["Status"] == "Available").sum())
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Available", available_count)
    status_col2.metric("Missing", len(status_df) - available_count)
    st.dataframe(
        status_df,
        hide_index=True,
        width="stretch",
    )

    history = st.session_state.get("admin_pipeline_history", [])
    if history:
        st.subheader("Recent Pipeline Runs")
        history_rows = [
            {
                "Command": result["command"],
                "Started": result["started_at"],
                "Finished": result["finished_at"],
                "Status": result["status"],
                "Duration (s)": result["duration_seconds"],
            }
            for result in history
        ]
        st.dataframe(
            pd.DataFrame(history_rows),
            hide_index=True,
            width="stretch",
        )
