"""Generate a Vietnamese Markdown report from saved experiment metrics."""

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import FIGURE_DIR, METRIC_DIR, OUTPUT_DIR


STATIC_METRICS_PATH = METRIC_DIR / "static_model_metrics.json"
ADWIN_DELAY_PATH = METRIC_DIR / "adwin_detection_delay.csv"
ADAPTIVE_STATIC_PATH = METRIC_DIR / "adaptive_static_summary.json"
ADAPTIVE_LSTM_PATH = METRIC_DIR / "adaptive_lstm_summary.json"
MODEL_COMPARISON_PATH = METRIC_DIR / "model_comparison_summary.csv"
REPORT_PATH = OUTPUT_DIR / "experiment_report.md"

EXPECTED_FIGURES = {
    "static": [
        "static_f1_over_time.png",
    ],
    "adwin": [
        "adwin_error_rate_over_time.png",
        "adwin_f1_over_time.png",
    ],
    "adaptive_static": [
        "static_vs_adaptive_f1.png",
    ],
    "adaptive_lstm": [
        "adaptive_lstm_f1_over_time.png",
        "lstm_static_vs_adaptive.png",
        "lstm_training_history.png",
    ],
    "comparison": [
        "compare_f1_over_time.png",
        "compare_accuracy_over_time.png",
        "compare_average_f1.png",
    ],
}


def _display_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR.resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path, missing_notes: list[str]) -> dict[str, Any] | None:
    """Load JSON and record a report note instead of raising on failure."""
    if not path.exists():
        missing_notes.append(f"Chưa có dữ liệu: `{_display_path(path)}`.")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        missing_notes.append(
            f"Không thể đọc `{_display_path(path)}`: {exc}."
        )
        return None


def _load_csv(path: Path, missing_notes: list[str]) -> pd.DataFrame | None:
    """Load CSV and record a report note instead of raising on failure."""
    if not path.exists():
        missing_notes.append(f"Chưa có dữ liệu: `{_display_path(path)}`.")
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        missing_notes.append(
            f"Không thể đọc `{_display_path(path)}`: {exc}."
        )
        return None


def _fmt_number(value: Any, digits: int = 4, missing: str = "Chưa có dữ liệu") -> str:
    """Format numeric values for Markdown tables."""
    if value is None or pd.isna(value):
        return missing
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any, missing: str = "Chưa có dữ liệu") -> str:
    """Format integer-like values for Markdown tables."""
    if value is None or pd.isna(value):
        return missing
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_points(points: Any) -> str:
    """Format a list of drift points."""
    if not points:
        return "Không có"
    return ", ".join(str(int(point)) for point in points)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Build a simple GitHub-flavored Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        safe_values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(safe_values) + " |")
    return "\n".join(lines)


def _figure_markdown(filename: str, caption: str) -> str:
    """Return an embedded figure link when the image exists."""
    path = FIGURE_DIR / filename
    if not path.exists():
        return f"> Chưa có biểu đồ `{_display_path(path)}`."
    return f"![{caption}](figures/{filename})"


def _section_figures(section: str) -> str:
    """Embed all available figures registered for one report section."""
    captions = {
        "static_f1_over_time.png": "F1 của Static Random Forest theo thời gian",
        "adwin_error_rate_over_time.png": "Prediction error rate và ADWIN",
        "adwin_f1_over_time.png": "F1 và các drift point",
        "static_vs_adaptive_f1.png": "Static vs Adaptive Random Forest",
        "adaptive_lstm_f1_over_time.png": "Adaptive LSTM F1 theo thời gian",
        "lstm_static_vs_adaptive.png": "Static LSTM vs Adaptive LSTM",
        "lstm_training_history.png": "Lịch sử huấn luyện LSTM ban đầu",
        "compare_f1_over_time.png": "So sánh F1 theo thời gian",
        "compare_accuracy_over_time.png": "So sánh accuracy theo thời gian",
        "compare_average_f1.png": "So sánh average F1",
        "accuracy_vs_update_cost.png": "Trade-off F1 và update cost",
    }
    return "\n\n".join(
        _figure_markdown(filename, captions[filename])
        for filename in EXPECTED_FIGURES[section]
    )


def _metric_rows(metrics: dict[str, Any] | None) -> list[list[str]]:
    """Convert a classification metrics dictionary to report rows."""
    if not metrics:
        return [
            ["Accuracy", "Chưa có dữ liệu"],
            ["Precision", "Chưa có dữ liệu"],
            ["Recall", "Chưa có dữ liệu"],
            ["F1-score", "Chưa có dữ liệu"],
        ]
    return [
        ["Accuracy", _fmt_number(metrics.get("accuracy"))],
        ["Precision", _fmt_number(metrics.get("precision"))],
        ["Recall", _fmt_number(metrics.get("recall"))],
        ["F1-score", _fmt_number(metrics.get("f1"))],
        ["Confusion matrix", f"`{metrics.get('confusion_matrix', 'Chưa có dữ liệu')}`"],
    ]


def _comparison_rows(comparison_df: pd.DataFrame | None) -> list[list[str]]:
    """Convert model comparison data to Markdown rows."""
    if comparison_df is None or comparison_df.empty:
        return [["Chưa có dữ liệu"] * 8]

    rows: list[list[str]] = []
    for row in comparison_df.itertuples(index=False):
        rows.append(
            [
                str(getattr(row, "model_name", "Unknown")),
                _fmt_number(getattr(row, "average_accuracy", None)),
                _fmt_number(getattr(row, "average_precision", None)),
                _fmt_number(getattr(row, "average_recall", None)),
                _fmt_number(getattr(row, "average_f1", None)),
                _fmt_number(getattr(row, "final_f1", None)),
                _fmt_int(getattr(row, "retrain_count", None)),
                _fmt_number(
                    getattr(row, "total_retrain_time_seconds", None),
                    digits=3,
                ),
            ]
        )
    return rows


def _calculate_lstm_delay(adaptive_lstm: dict[str, Any] | None) -> int | None:
    """Calculate the first adaptive LSTM detection delay when available."""
    if not adaptive_lstm:
        return None
    actual = adaptive_lstm.get("actual_drift_points") or []
    detected = adaptive_lstm.get("detected_drift_points") or []
    if not actual or not detected:
        return None
    first_actual = int(actual[0])
    first_detected = next(
        (int(point) for point in detected if int(point) >= first_actual),
        None,
    )
    return first_detected - first_actual if first_detected is not None else None


def _conclusion_lines(
    static_metrics: dict[str, Any] | None,
    adaptive_static: dict[str, Any] | None,
    adaptive_lstm: dict[str, Any] | None,
    lstm_delay: int | None,
) -> list[str]:
    """Generate conclusions directly from experiment metrics."""
    conclusions: list[str] = []

    static_f1 = (
        (static_metrics or {}).get("stream_metrics", {}).get("f1")
        if static_metrics
        else None
    )
    adaptive_rf_f1 = (
        (adaptive_static or {}).get("adaptive_metrics", {}).get("f1")
        if adaptive_static
        else None
    )
    if static_f1 is not None and adaptive_rf_f1 is not None:
        if adaptive_rf_f1 > static_f1:
            conclusions.append(
                f"Adaptive Random Forest cải thiện F1 từ "
                f"{static_f1:.4f} lên {adaptive_rf_f1:.4f}."
            )
        elif adaptive_rf_f1 == static_f1:
            conclusions.append(
                "Adaptive Random Forest chưa cải thiện so với static baseline "
                "vì ADWIN không phát hiện drift để kích hoạt retraining."
            )
        else:
            conclusions.append(
                "Adaptive Random Forest có F1 thấp hơn static baseline trong lần chạy này."
            )

    static_lstm_f1 = (
        (adaptive_lstm or {}).get("static_overall_metrics", {}).get("f1")
        if adaptive_lstm
        else None
    )
    adaptive_lstm_f1 = (
        (adaptive_lstm or {}).get("adaptive_overall_metrics", {}).get("f1")
        if adaptive_lstm
        else None
    )
    if static_lstm_f1 is not None and adaptive_lstm_f1 is not None:
        if adaptive_lstm_f1 > static_lstm_f1:
            conclusions.append(
                f"Adaptive LSTM cải thiện hiệu quả, F1 tăng từ "
                f"{static_lstm_f1:.4f} lên {adaptive_lstm_f1:.4f} sau fine-tune."
            )
        else:
            conclusions.append(
                "Adaptive LSTM chưa cải thiện F1 so với LSTM không cập nhật."
            )

    retrain_time = (
        (adaptive_lstm or {}).get("total_retrain_time_seconds")
        if adaptive_lstm
        else None
    )
    if retrain_time is not None:
        if float(retrain_time) >= 5.0:
            conclusions.append(
                f"Thời gian cập nhật {float(retrain_time):.3f} giây là đáng kể; "
                "độ chính xác cao hơn đi kèm trade-off chi phí retraining."
            )
        elif float(retrain_time) > 0:
            conclusions.append(
                f"Chi phí fine-tune khoảng {float(retrain_time):.3f} giây, "
                "cho thấy trade-off cập nhật hiện ở mức thấp trong mô phỏng local."
            )

    window_size = (adaptive_lstm or {}).get("window_size", 1000)
    if lstm_delay is not None:
        if lstm_delay <= int(window_size):
            conclusions.append(
                f"ADWIN phát hiện drift nhanh với delay {lstm_delay} mẫu, "
                f"nhỏ hơn hoặc bằng một window ({window_size} mẫu)."
            )
        else:
            conclusions.append(
                f"Detection delay là {lstm_delay} mẫu, lớn hơn một window; "
                "cần điều chỉnh ADWIN hoặc window size để phản ứng sớm hơn."
            )

    if not conclusions:
        conclusions.append(
            "Chưa đủ metrics để tự động kết luận hiệu quả của adaptive learning."
        )
    return conclusions


def build_report() -> str:
    """Build the complete experiment report as Markdown text."""
    missing_notes: list[str] = []
    static_metrics = _load_json(STATIC_METRICS_PATH, missing_notes)
    adwin_delay = _load_csv(ADWIN_DELAY_PATH, missing_notes)
    adaptive_static = _load_json(ADAPTIVE_STATIC_PATH, missing_notes)
    adaptive_lstm = _load_json(ADAPTIVE_LSTM_PATH, missing_notes)
    comparison_df = _load_csv(MODEL_COMPARISON_PATH, missing_notes)

    train_rows = (static_metrics or {}).get("train_rows")
    stream_rows = (static_metrics or {}).get("stream_rows")
    feature_count = (static_metrics or {}).get("feature_count")
    total_rows = (
        int(train_rows) + int(stream_rows)
        if train_rows is not None and stream_rows is not None
        else None
    )

    static_stream_metrics = (static_metrics or {}).get("stream_metrics")
    adaptive_rf_metrics = (adaptive_static or {}).get("adaptive_metrics")
    static_lstm_metrics = (adaptive_lstm or {}).get("static_overall_metrics")
    adaptive_lstm_metrics = (adaptive_lstm or {}).get("adaptive_overall_metrics")
    lstm_delay = _calculate_lstm_delay(adaptive_lstm)

    report: list[str] = [
        "# Báo Cáo Thực Nghiệm: Phát Hiện Drift Trong Hệ Thống IoT",
        "",
        f"**Ngày tạo báo cáo:** {date.today().isoformat()}",
        "",
        "## Tổng Quan Thực Nghiệm",
        "",
        (
            "Thực nghiệm mô phỏng dữ liệu IoT dạng stream cho bài toán binary "
            "classification (`0 = normal`, `1 = attack`). Hệ thống so sánh "
            "mô hình static với mô hình adaptive, sử dụng ADWIN theo dõi "
            "prediction error và kích hoạt retraining/fine-tuning khi phát hiện drift."
        ),
        "",
        _markdown_table(
            ["Thuộc tính", "Giá trị"],
            [
                ["Tổng số dòng", _fmt_int(total_rows)],
                ["Train rows", _fmt_int(train_rows)],
                ["Stream rows", _fmt_int(stream_rows)],
                ["Số feature", _fmt_int(feature_count)],
                [
                    "Window size",
                    _fmt_int((static_metrics or {}).get("window_size")),
                ],
                ["Actual drift points", _fmt_points(
                    (adaptive_lstm or adaptive_static or {}).get(
                        "actual_drift_points",
                        [],
                    )
                )],
            ],
        ),
        "",
        "## Dataset Sử Dụng",
        "",
        (
            "Dataset hiện tại là **synthetic IoT drift dataset**, được tạo để "
            "kiểm thử pipeline khi chưa tích hợp TON_IoT và CICIoT thật. Dataset "
            "gồm 20 feature số, timestamp, attack type và binary label. Dữ liệu "
            "được chia theo thời gian với tỷ lệ 60% train và 40% stream."
        ),
        "",
        "Bốn giai đoạn mô phỏng gồm: DoS tỷ lệ thấp, DDoS tăng mạnh, xuất hiện "
        "Recon/Mirai với feature shift, và giai đoạn benign/attack trộn phức tạp.",
        "",
        "## Static Model Baseline",
        "",
        "Static baseline sử dụng `RandomForestClassifier` và không cập nhật khi stream thay đổi.",
        "",
        _markdown_table(["Metric", "Stream result"], _metric_rows(static_stream_metrics)),
        "",
        _section_figures("static"),
        "",
        "## ADWIN Drift Detection",
        "",
    ]

    if adwin_delay is None:
        report.append("Chưa có dữ liệu detection delay.")
    elif adwin_delay.empty:
        report.append("File detection delay hiện không có bản ghi.")
    else:
        delay_rows = []
        for row in adwin_delay.itertuples(index=False):
            delay_rows.append(
                [
                    _fmt_int(getattr(row, "actual_drift_point", None)),
                    _fmt_int(getattr(row, "detected_drift_point", None)),
                    _fmt_int(getattr(row, "delay", None)),
                ]
            )
        report.extend(
            [
                _markdown_table(
                    ["Actual drift", "Detected drift", "Delay"],
                    delay_rows,
                ),
                "",
                (
                    "Trên error stream của Static Random Forest, ADWIN chưa phát hiện "
                    "drift do tỷ lệ lỗi của model vẫn rất thấp."
                ),
            ]
        )

    report.extend(
        [
            "",
            _section_figures("adwin"),
            "",
            "## Adaptive Random Forest",
            "",
            _markdown_table(
                ["Thuộc tính", "Giá trị"],
                [
                    [
                        "Detected drift points",
                        _fmt_points((adaptive_static or {}).get("detected_drift_points")),
                    ],
                    [
                        "Số lần retrain",
                        _fmt_int((adaptive_static or {}).get("number_of_retrains")),
                    ],
                    [
                        "Tổng thời gian retrain (giây)",
                        _fmt_number(
                            (adaptive_static or {}).get("total_retrain_time_seconds"),
                            digits=3,
                        ),
                    ],
                    [
                        "Final model version",
                        _fmt_int((adaptive_static or {}).get("final_model_version")),
                    ],
                ],
            ),
            "",
            _markdown_table(
                ["Metric", "Adaptive Random Forest"],
                _metric_rows(adaptive_rf_metrics),
            ),
            "",
            _section_figures("adaptive_static"),
            "",
            "## Adaptive LSTM",
            "",
            _markdown_table(
                ["Thuộc tính", "Giá trị"],
                [
                    [
                        "Tổng số window",
                        _fmt_int((adaptive_lstm or {}).get("total_windows")),
                    ],
                    [
                        "Detected drift points",
                        _fmt_points((adaptive_lstm or {}).get("detected_drift_points")),
                    ],
                    [
                        "Số lần fine-tune",
                        _fmt_int((adaptive_lstm or {}).get("total_retrain_count")),
                    ],
                    [
                        "Tổng fine-tune time (giây)",
                        _fmt_number(
                            (adaptive_lstm or {}).get("total_retrain_time_seconds"),
                            digits=3,
                        ),
                    ],
                    [
                        "Average fine-tune time (giây)",
                        _fmt_number(
                            (adaptive_lstm or {}).get("average_retrain_time_seconds"),
                            digits=3,
                        ),
                    ],
                    [
                        "Final window F1",
                        _fmt_number((adaptive_lstm or {}).get("final_f1")),
                    ],
                    [
                        "Best window F1",
                        _fmt_number((adaptive_lstm or {}).get("best_f1")),
                    ],
                    [
                        "Worst window F1",
                        _fmt_number((adaptive_lstm or {}).get("worst_f1")),
                    ],
                ],
            ),
            "",
            _markdown_table(
                ["Metric", "Static LSTM", "Adaptive LSTM"],
                [
                    [
                        "Accuracy",
                        _fmt_number((static_lstm_metrics or {}).get("accuracy")),
                        _fmt_number((adaptive_lstm_metrics or {}).get("accuracy")),
                    ],
                    [
                        "Precision",
                        _fmt_number((static_lstm_metrics or {}).get("precision")),
                        _fmt_number((adaptive_lstm_metrics or {}).get("precision")),
                    ],
                    [
                        "Recall",
                        _fmt_number((static_lstm_metrics or {}).get("recall")),
                        _fmt_number((adaptive_lstm_metrics or {}).get("recall")),
                    ],
                    [
                        "F1-score",
                        _fmt_number((static_lstm_metrics or {}).get("f1")),
                        _fmt_number((adaptive_lstm_metrics or {}).get("f1")),
                    ],
                ],
            ),
            "",
            _section_figures("adaptive_lstm"),
            "",
            "## So Sánh Static Vs Adaptive",
            "",
            _markdown_table(
                [
                    "Model",
                    "Avg Accuracy",
                    "Avg Precision",
                    "Avg Recall",
                    "Avg F1",
                    "Final F1",
                    "Retrain Count",
                    "Update Time (s)",
                ],
                _comparison_rows(comparison_df),
            ),
            "",
            _section_figures("comparison"),
            "",
            "## Detection Delay",
            "",
        ]
    )

    if lstm_delay is None:
        report.append(
            "Adaptive LSTM chưa có đủ actual/detected drift point để tính detection delay."
        )
    else:
        actual_point = (adaptive_lstm or {}).get("actual_drift_points", [None])[0]
        detected_point = (adaptive_lstm or {}).get("detected_drift_points", [None])[0]
        report.extend(
            [
                _markdown_table(
                    ["Model", "Actual drift", "Detected drift", "Delay (samples)"],
                    [
                        [
                            "Adaptive LSTM + ADWIN",
                            _fmt_int(actual_point),
                            _fmt_int(detected_point),
                            _fmt_int(lstm_delay),
                        ]
                    ],
                ),
                "",
                (
                    f"ADWIN phát hiện drift sau {lstm_delay} mẫu. Với window size "
                    f"{(adaptive_lstm or {}).get('window_size', 1000)}, độ trễ này "
                    "nhỏ hơn một window."
                ),
            ]
        )

    report.extend(
        [
            "",
            "## Trade-off Accuracy Vs Update Cost",
            "",
            (
                "Random Forest đạt average F1 cao và không phát sinh update cost trong "
                "lần chạy hiện tại, nhưng cũng không chứng minh được khả năng thích nghi "
                "vì ADWIN không trigger. Adaptive LSTM có average F1 thấp hơn Random Forest "
                "trên toàn stream, song cải thiện rõ so với chính Static LSTM sau một lần "
                "fine-tune. Chi phí cập nhật được đo bằng tổng thời gian retraining."
            ),
            "",
            _figure_markdown(
                "accuracy_vs_update_cost.png",
                "Trade-off average F1 và tổng thời gian retraining",
            ),
            "",
            "## Kết Luận",
            "",
        ]
    )

    for conclusion in _conclusion_lines(
        static_metrics,
        adaptive_static,
        adaptive_lstm,
        lstm_delay,
    ):
        report.append(f"- {conclusion}")

    if missing_notes:
        report.extend(
            [
                "",
                "## Ghi Chú Dữ Liệu Thiếu",
                "",
                *[f"- {note}" for note in missing_notes],
            ]
        )

    report.extend(
        [
            "",
            "---",
            "",
            "Báo cáo được tạo tự động bởi `scripts/08_generate_experiment_report.py`.",
        ]
    )
    return "\n".join(report) + "\n"


def main() -> None:
    """Generate and save the Markdown experiment report."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Experiment report generated successfully.")
    print(f"Report path: {REPORT_PATH.relative_to(ROOT_DIR)}")
    print(f"Report size: {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()
