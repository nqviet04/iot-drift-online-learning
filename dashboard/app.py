"""Streamlit dashboard for IoT drift-detection experiment results.

Run from the project root:
    streamlit run dashboard/app.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CLOUD_MODEL_DIR, DATA_SYNTHETIC_DIR, METRIC_DIR


SYNTHETIC_DATA_PATH = DATA_SYNTHETIC_DIR / "synthetic_iot_drift.csv"
STATIC_WINDOW_PATH = METRIC_DIR / "static_window_metrics.csv"
ADWIN_DRIFTS_PATH = METRIC_DIR / "adwin_detected_drifts.json"
ADWIN_DELAY_PATH = METRIC_DIR / "adwin_detection_delay.csv"
ADAPTIVE_STATIC_WINDOW_PATH = METRIC_DIR / "adaptive_static_window_metrics.csv"
ADAPTIVE_STATIC_RETRAIN_PATH = METRIC_DIR / "adaptive_static_retrain_log.csv"
ADAPTIVE_LSTM_WINDOW_PATH = METRIC_DIR / "adaptive_lstm_window_metrics.csv"
ADAPTIVE_LSTM_RETRAIN_PATH = METRIC_DIR / "adaptive_lstm_retrain_log.csv"
MODEL_COMPARISON_PATH = METRIC_DIR / "model_comparison_summary.csv"

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

dataset_tab, static_tab, adwin_tab, adaptive_tab, comparison_tab = st.tabs(
    [
        "Dataset Overview",
        "Static Model Performance",
        "ADWIN Drift Detection",
        "Adaptive Model",
        "Model Comparison",
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
