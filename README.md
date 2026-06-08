# IoT Drift Online Learning

Đồ án môn học: **Phát hiện drift trong hệ thống IoT bằng Online Learning**.

Project mô phỏng luồng dữ liệu IoT, huấn luyện mô hình binary classification (`0 = benign/normal`, `1 = attack`), phát hiện concept drift bằng ADWIN và so sánh static model với adaptive model. Khi chưa có TON_IoT hoặc CICIoT thật, pipeline sẽ dùng synthetic dataset để kiểm thử end-to-end.

## Mục Tiêu

- Dùng TON_IoT chia theo thời gian để mô phỏng IoT stream.
- Dùng CICIoT để mô phỏng concept drift, ví dụ thay đổi pattern tấn công hoặc tỷ lệ attack/benign.
- So sánh static model và adaptive model.
- Dùng ADWIN để phát hiện suy giảm hiệu năng.
- Xây dựng LSTM online/fine-tuning theo window.
- Khi phát hiện drift, adaptive model retrain hoặc fine-tune.
- Mô phỏng cloud model storage bằng `cloud_model_storage/`.
- Lưu kết quả experiment vào `outputs/`.

## Cấu Trúc Project

```text
iot-drift-online-learning/
├── api/
├── cloud_model_storage/
├── dashboard/
├── data/
│   ├── processed/
│   ├── raw/
│   │   ├── CICIoT/
│   │   └── TON_IoT/
│   └── synthetic/
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── logs/
│   └── metrics/
├── scripts/
└── src/
```

## Cài Đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Trên macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quy Ước Dữ Liệu Và Artifact

- Không commit dataset thật vào GitHub.
- Đặt TON_IoT tại `data/raw/TON_IoT/`.
- Đặt CICIoT tại `data/raw/CICIoT/`.
- Dữ liệu đã xử lý lưu ở `data/processed/`.
- Synthetic CSV sinh ra để test pipeline lưu ở `data/synthetic/`.
- Metrics, figures và logs lưu ở `outputs/`.
- Model artifact lưu ở `cloud_model_storage/`.
- Không commit `.env`, AWS key, secret, token, dataset, model, output hoặc log.

## Pipeline Dự Kiến

1. `scripts/00_generate_synthetic_data.py`: tạo synthetic dataset để test.
2. `scripts/01_prepare_data.py`: chuẩn hóa dữ liệu và tạo train/stream windows.
3. `scripts/02_train_static.py`: train baseline static model.
4. `scripts/03_run_stream_adwin.py`: chạy stream và ADWIN drift detection.
5. `scripts/04_run_adaptive_static.py`: adaptive retraining với model truyền thống.
6. `scripts/05_train_lstm.py`: train LSTM ban đầu.
7. `scripts/06_run_adaptive_lstm.py`: fine-tune LSTM theo window khi drift.
8. `scripts/07_compare_models.py`: so sánh static/adaptive.
9. `scripts/08_generate_experiment_report.py`: tạo báo cáo kết quả.

## Metrics Báo Cáo

- Accuracy
- Precision
- Recall
- F1-score
- Detection delay
- Số lần retrain/fine-tune
- Thời gian retrain/fine-tune
- Trade-off accuracy vs update cost

## Trạng Thái

Skeleton project đã được tạo. Các module hiện là khung ban đầu để phát triển pipeline ở các bước tiếp theo.
