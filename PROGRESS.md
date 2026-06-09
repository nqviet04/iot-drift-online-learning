# Project Progress

## 2026-06-10

## Done

### Project foundation

- Hoàn thành skeleton project theo cấu trúc module hóa.
- Cấu hình đường dẫn tập trung bằng `pathlib` trong `src/config.py`.
- Tự tạo các thư mục data, output và model storage khi import config.
- Thêm `.gitignore`, `.env.example`, `requirements.txt` và README.
- Không hard-code đường dẫn tuyệt đối, AWS key, token hoặc secret.
- Dataset, model, output, log và `.env` đã được chặn khỏi Git.

### Synthetic IoT data

- Hoàn thành `src/drift_simulator.py`.
- Hoàn thành `scripts/00_generate_synthetic_data.py`.
- Sinh dataset gồm 50.000 dòng, 20 numeric features, `timestamp`, `attack_type` và `label_binary`.
- Mô phỏng 4 giai đoạn concept drift.
- Actual drift points: `[12500, 25000, 37500]`.

### Data loading and preprocessing

- Hoàn thành `src/data_loader.py`.
- Hỗ trợ synthetic, TON_IoT, CICIoT và time-based split không shuffle.
- Hoàn thành `src/preprocessing.py`.
- Tự phát hiện label, tạo binary label, loại metadata, xử lý missing/inf và chọn numeric features.
- `StandardScaler` chỉ fit trên train set để tránh data leakage.
- Preprocessor hỗ trợ save/load bằng Joblib.

### Static model

- Hoàn thành `src/static_model.py`.
- Hỗ trợ Random Forest và SGD classifier.
- Hoàn thành `scripts/02_train_static.py`.
- Random Forest stream metrics:
  - Accuracy: `0.9960`
  - Precision: `0.9991`
  - Recall: `0.9920`
  - F1: `0.9955`

### Evaluation

- Hoàn thành `src/evaluation.py`.
- Hỗ trợ classification metrics, evaluation theo window, JSON/CSV output và biểu đồ theo thời gian.
- Static model được đánh giá trên 20 stream windows, mỗi window 1.000 dòng.

### ADWIN drift detection

- Hoàn thành `src/adwin_detector.py`.
- Hoàn thành `scripts/03_run_stream_adwin.py`.
- ADWIN theo dõi prediction error stream và tính detection delay.
- Random Forest không phát hiện drift trên synthetic stream hiện tại do error rate chỉ khoảng `0.00395`.
- ADWIN wrapper đã được kiểm thử riêng với abrupt error drift và phát hiện thành công với delay 23 mẫu.

### Adaptive Random Forest

- Hoàn thành `src/adaptive_trainer.py`.
- Có FIFO recent buffer, versioned retraining và retrain log.
- Hoàn thành `scripts/04_run_adaptive_static.py`.
- Trên experiment hiện tại ADWIN không trigger, vì vậy:
  - Retrain count: `0`
  - Final model version: `0`
  - Static F1 và adaptive F1 đều khoảng `0.9955`
- Nhánh retrain/versioning đã được smoke test riêng và hoạt động đúng.

### Initial LSTM

- Hoàn thành `src/lstm_model.py`.
- Hỗ trợ sequence creation, build/train/predict và save/load Keras model.
- Hoàn thành `scripts/05_train_lstm.py`.
- Early stopping dừng sau 3/5 epoch và khôi phục trọng số tốt nhất.
- Initial LSTM stream metrics:
  - Accuracy: `0.9568`
  - Precision: `0.9993`
  - Recall: `0.9034`
  - F1: `0.9489`

### Adaptive LSTM

- Hoàn thành `scripts/06_run_adaptive_lstm.py`.
- ADWIN phát hiện drift tại index `38016`, sau actual drift `37500`.
- Detection delay: `516` mẫu.
- Fine-tune một lần trên recent buffer 5.000 dòng:
  - Epochs: `3`
  - Batch size: `64`
  - Fine-tune time gần nhất: khoảng `1.83` giây
  - Final model version: `1`
- Kết quả:
  - Static LSTM overall F1: khoảng `0.9492`
  - Adaptive LSTM overall F1: khoảng `0.9573`
  - Final window F1: khoảng `0.9987`
  - Best window F1: khoảng `0.9987`
  - Worst window F1: khoảng `0.8899`

### Model comparison

- Hoàn thành `scripts/07_compare_models.py`.
- Tổng hợp Static Random Forest, Adaptive Random Forest và Adaptive LSTM.
- Script warning nhưng không crash khi thiếu input file.
- Average F1 gần nhất:
  - Static Random Forest: `0.9954`
  - Adaptive Random Forest: `0.9954`
  - Adaptive LSTM: `0.9584`
- Đã tạo biểu đồ accuracy/F1 theo thời gian và trade-off F1 vs update cost.

### FastAPI model endpoint

- Hoàn thành `api/main.py`.
- API tự load model theo thứ tự:
  1. Adaptive LSTM version mới nhất.
  2. Adaptive Random Forest version mới nhất.
  3. Static Random Forest.
- Model active hiện tại: `Adaptive LSTM`, version `1`.
- Hoàn thành các endpoint:
  - `GET /`
  - `GET /health`
  - `GET /models`
  - `POST /predict`
  - `POST /predict_batch`
- Có Pydantic schema, feature validation, confidence và lỗi 503 nếu không có model.
- README đã có hướng dẫn chạy API và JSON request mẫu.

## Verified Commands

Các command sau đã chạy thành công:

```bash
python scripts/00_generate_synthetic_data.py
python scripts/02_train_static.py
python scripts/03_run_stream_adwin.py
python scripts/04_run_adaptive_static.py
python scripts/05_train_lstm.py
python scripts/06_run_adaptive_lstm.py
python scripts/07_compare_models.py
uvicorn api.main:app --reload
```

Các kiểm tra bổ sung đã pass:

- Python syntax check bằng `python -m compileall`.
- Load lại Joblib preprocessor/model.
- Load lại `.keras` model và chạy prediction.
- Smoke test RecentBuffer FIFO và adaptive model versioning.
- Smoke test ADWIN trên abrupt error stream.
- FastAPI startup load đúng `lstm_adaptive_v1.keras`.
- HTTP `/health`, `/models` và `/predict` trả response hợp lệ.

## Generated Outputs

### Synthetic data

- `data/synthetic/synthetic_iot_drift.csv`

### Model artifacts

- `cloud_model_storage/static_random_forest.joblib`
- `cloud_model_storage/preprocessor.joblib`
- `cloud_model_storage/adaptive_rf_v0.joblib`
- `cloud_model_storage/lstm_initial.keras`
- `cloud_model_storage/lstm_preprocessor.joblib`
- `cloud_model_storage/lstm_adaptive_v1.keras`

### Metrics and logs

- `outputs/metrics/synthetic_drift_points.json`
- `outputs/metrics/static_model_metrics.json`
- `outputs/metrics/static_window_metrics.csv`
- `outputs/metrics/adwin_detected_drifts.json`
- `outputs/metrics/adwin_detection_delay.csv`
- `outputs/metrics/adaptive_static_window_metrics.csv`
- `outputs/metrics/adaptive_static_retrain_log.csv`
- `outputs/metrics/adaptive_static_retrain_log.json`
- `outputs/metrics/adaptive_static_summary.json`
- `outputs/metrics/lstm_initial_metrics.json`
- `outputs/metrics/adaptive_lstm_window_metrics.csv`
- `outputs/metrics/adaptive_lstm_retrain_log.csv`
- `outputs/metrics/adaptive_lstm_detected_drifts.json`
- `outputs/metrics/adaptive_lstm_summary.json`
- `outputs/metrics/model_comparison_summary.csv`
- `outputs/metrics/model_comparison_summary.json`

### Figures

- `outputs/figures/static_f1_over_time.png`
- `outputs/figures/adwin_error_rate_over_time.png`
- `outputs/figures/adwin_f1_over_time.png`
- `outputs/figures/static_vs_adaptive_f1.png`
- `outputs/figures/lstm_training_history.png`
- `outputs/figures/adaptive_lstm_f1_over_time.png`
- `outputs/figures/lstm_static_vs_adaptive.png`
- `outputs/figures/compare_f1_over_time.png`
- `outputs/figures/compare_accuracy_over_time.png`
- `outputs/figures/compare_average_f1.png`
- `outputs/figures/accuracy_vs_update_cost.png`

Các generated artifacts trên nằm trong đường dẫn đã được `.gitignore`.

## Known Issues

- `scripts/01_prepare_data.py` vẫn là placeholder.
- `scripts/08_generate_experiment_report.py` vẫn là placeholder.
- Chưa chạy pipeline với dataset TON_IoT và CICIoT thật.
- Mapping schema/label cụ thể cho từng phiên bản TON_IoT và CICIoT chưa được xác nhận.
- Synthetic data hiện khá dễ đối với Random Forest, làm prediction error quá thấp để ADWIN trigger adaptive static retraining.
- Adaptive LSTM tạo sequence riêng theo từng window, nên context giữa hai window chưa được giữ liên tục.
- API với LSTM phải pad/lặp dữ liệu khi request không có đủ 10 timesteps; đây là mô phỏng, chưa phải stateful production inference.
- Dashboard Streamlit vẫn là giao diện placeholder.
- AWS S3 upload/download chưa được triển khai; project hiện chỉ dùng local cloud model storage.
- Chưa có automated unit/integration test suite.
- Dependencies chưa pin version, nên khả năng tái lập môi trường giữa các máy chưa được đảm bảo hoàn toàn.
- TensorFlow trên native Windows hiện chạy CPU; không sử dụng GPU với TensorFlow phiên bản mới.

## Next Steps

1. Implement `scripts/01_prepare_data.py` cho synthetic, TON_IoT và CICIoT.
2. Tải và kiểm thử pipeline với dataset TON_IoT/CICIoT thật.
3. Chuẩn hóa schema mapping, timestamp và binary label cho từng dataset.
4. Implement `scripts/08_generate_experiment_report.py`.
5. Tạo báo cáo tự động gồm metrics, detection delay, retrain cost và các figure.
6. Cải thiện synthetic drift để static model suy giảm rõ hơn và ADWIN có thể trigger.
7. Giữ sequence context xuyên qua các stream window cho adaptive LSTM.
8. Thêm test suite cho data loader, preprocessing, ADWIN, model registry và API.
9. Hoàn thiện Streamlit dashboard.
10. Pin dependency versions và thêm hướng dẫn tái lập môi trường.
11. Mở rộng `aws_storage.py` để upload/download model qua S3 bằng environment variables hoặc IAM role.
