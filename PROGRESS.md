# Progress

## 2026-06-09

### Đã hoàn thành

- Tạo skeleton project theo cấu trúc yêu cầu.
- Thêm `.gitignore` an toàn cho Python ML project.
- Thêm `.env.example` và không tạo `.env`.
- Thêm `.gitkeep` để giữ các thư mục rỗng cần thiết.
- Thêm `requirements.txt` với các thư viện chính: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `river`, `tensorflow`, `joblib`, `fastapi`, `uvicorn`, `streamlit`, `plotly`, `tqdm`, `boto3`, `python-dotenv`.
- Thêm README bản đầu tiên.
- Tạo cấu hình tập trung trong `src/config.py` bằng `pathlib`, không hard-code đường dẫn tuyệt đối.
- Cấu hình AWS chỉ đọc từ biến môi trường, không hard-code credential.
- Khi import `src.config`, tự tạo các thư mục cần thiết nếu chưa tồn tại.

### Synthetic Data

- Implement `src/drift_simulator.py`.
- Implement `scripts/00_generate_synthetic_data.py`.
- Sinh synthetic dataset mặc định `50000` dòng, `20` feature số (`feature_0` đến `feature_19`).
- Dataset có các cột `timestamp`, `attack_type`, `label_binary`.
- Mô phỏng 4 giai đoạn drift:
  - Giai đoạn 1: benign nhiều, attack ít, chủ yếu DoS.
  - Giai đoạn 2: tăng tỷ lệ attack, chuyển sang DDoS.
  - Giai đoạn 3: xuất hiện Recon/Mirai, phân phối feature thay đổi mạnh.
  - Giai đoạn 4: benign và attack trộn phức tạp hơn.
- Lưu synthetic CSV vào `data/synthetic/synthetic_iot_drift.csv`.
- Lưu drift points thật vào `outputs/metrics/synthetic_drift_points.json`.
- Drift points hiện tại: `[12500, 25000, 37500]`.
- Các file CSV/output sinh ra đã được `.gitignore` chặn, an toàn cho GitHub.

### Data Loading

- Implement `src/data_loader.py`.
- Có các hàm:
  - `load_csv_dataset(path)`
  - `load_synthetic_dataset()`
  - `load_ton_iot_dataset(filename)`
  - `load_cic_iot_dataset(filename)`
  - `time_based_split(df, train_ratio=0.6, timestamp_col=None)`
- `load_synthetic_dataset()` tự sinh synthetic dataset nếu file chưa tồn tại.
- Load TON_IoT/CICIoT không giả định tên cột cố định.
- `time_based_split` không shuffle, phù hợp mô phỏng stream theo thời gian.

### Preprocessing

- Implement `src/preprocessing.py`.
- Có các hàm:
  - `detect_label_column(df)`
  - `create_binary_label(df, label_col=None)`
  - `clean_features(df, label_col="label_binary")`
- Implement class `Preprocessor` dùng `StandardScaler`.
- `Preprocessor` có:
  - `fit_transform(X_train)`
  - `transform(X)`
  - `save(path)`
  - `load(path)`
- Đã đảm bảo scaler chỉ fit trên train set, stream/test chỉ transform để tránh data leakage.
- Smoke test synthetic dataset thành công: `20` feature numeric, split `30000` train và `20000` stream.

### Static Baseline

- Implement `src/static_model.py`.
- Có các hàm:
  - `train_random_forest(X_train, y_train, random_state=42)`
  - `train_sgd_classifier(X_train, y_train, random_state=42)`
  - `save_model(model, path)`
  - `load_model(path)`
- Implement `scripts/02_train_static.py`.
- Script hiện thực hiện:
  - Load synthetic dataset.
  - Tạo `label_binary`.
  - Chia time-based split `60/40`.
  - Clean numeric features.
  - Fit preprocessor trên train set.
  - Train Random Forest static baseline.
  - Evaluate trên train và stream test.
  - Lưu model vào `cloud_model_storage/static_random_forest.joblib`.
  - Lưu preprocessor vào `cloud_model_storage/preprocessor.joblib`.
  - Lưu metrics vào `outputs/metrics/static_model_metrics.json`.

### Evaluation Theo Window

- Implement `src/evaluation.py`.
- Có các hàm:
  - `compute_classification_metrics(y_true, y_pred)`
  - `evaluate_by_windows(model, X_stream, y_stream, window_size=1000)`
  - `save_metrics_json(metrics, path)`
  - `save_window_metrics_csv(df, path)`
  - `plot_metric_over_time(window_metrics_df, metric_name, output_path, drift_points=None, detected_drifts=None)`
- Cập nhật `scripts/02_train_static.py` để evaluate stream theo từng window.
- Lưu window metrics vào `outputs/metrics/static_window_metrics.csv`.
- Lưu biểu đồ F1 theo thời gian vào `outputs/figures/static_f1_over_time.png`.
- Đã chạy thành công:

```bash
python scripts/02_train_static.py
```

### Kết quả Static Baseline Hiện Tại

- Train rows: `30000`.
- Stream test rows: `20000`.
- Feature count: `20`.
- Window size: `1000`.
- Stream-relative actual drift point: `[7500]`.

Train metrics:

- Accuracy: `1.0000`
- Precision: `1.0000`
- Recall: `1.0000`
- F1-score: `1.0000`
- Confusion matrix: `[[22745, 0], [0, 7255]]`

Stream test metrics:

- Accuracy: `0.9960`
- Precision: `0.9991`
- Recall: `0.9920`
- F1-score: `0.9955`
- Confusion matrix: `[[11118, 8], [71, 8803]]`

### Artifact Local Đã Sinh

- `data/synthetic/synthetic_iot_drift.csv`
- `outputs/metrics/synthetic_drift_points.json`
- `cloud_model_storage/static_random_forest.joblib`
- `cloud_model_storage/preprocessor.joblib`
- `outputs/metrics/static_model_metrics.json`
- `outputs/metrics/static_window_metrics.csv`
- `outputs/figures/static_f1_over_time.png`

Các artifact trên nằm trong thư mục đã được `.gitignore`, không đưa lên GitHub.

### Việc tiếp theo

- Implement `src/adwin_detector.py` đầy đủ.
- Implement `scripts/03_run_stream_adwin.py` để phát hiện drift trên stream.
- Implement adaptive retraining pipeline trong `src/adaptive_trainer.py`.
- Implement `scripts/04_run_adaptive_static.py`.
- Implement LSTM model và fine-tuning theo window.
- Implement `scripts/05_train_lstm.py` và `scripts/06_run_adaptive_lstm.py`.
- Implement so sánh static/adaptive trong `scripts/07_compare_models.py`.
- Implement report tổng hợp trong `scripts/08_generate_experiment_report.py`.
- Cập nhật README với hướng dẫn chạy pipeline từng bước.
