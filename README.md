# FraudGNN-RL Reproduction Project

Project này dựng lại pipeline **FraudGNN-RL** dựa trên mô tả trong paper:

> FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection

## Lưu ý khoa học quan trọng

Paper gốc chưa công bố source code chính thức trong PDF. Vì vậy project này là bản **reproduction from paper description**, tức là tái hiện gần nhất có thể theo thuật toán và thông số được mô tả, không thể cam kết giống 100% với mã nguồn nội bộ của tác giả.

Các điểm bám sát paper:

- Dữ liệu giao dịch được biến thành transaction graph.
- Node là transaction trong graph similarity-time.
- Edge được tạo khi transaction gần nhau theo thời gian và cosine similarity vượt ngưỡng.
- TSSGC gồm 3 thành phần:
  - Temporal modeling
  - Spatial modeling
  - Semantic modeling
- TSSGC mặc định 3 layers, hidden dimension 64.
- Classifier sinh fraud score.
- DQN agent điều chỉnh threshold dựa trên score stream.
- Metric gồm AUC-ROC, AUC-PR, F1, Recall@1%.

## Cấu trúc thư mục

```text
FRAUDGNN/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── graphs/
├── src/
│   ├── data/
│   ├── graph/
│   ├── models/
│   ├── train/
│   ├── eval/
│   └── utils/
├── scripts/
├── outputs/
└── tests/
```

## Cài đặt

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Nếu `torch-geometric` bị lỗi, cài theo hướng dẫn chính thức của PyTorch Geometric đúng với phiên bản CUDA/CPU của máy.

## Chuẩn bị dữ liệu

Đặt file dataset vào `data/raw/` và sửa đường dẫn trong file config.

Ví dụ PaySim:

```text
data/raw/paysim.csv
```

Config tương ứng:

```text
configs/paysim.yaml
```

Lưu ý PaySim dùng label thật là `isFraud`, không dùng `isFlaggedFraud`.

## Chạy reproduction

```bash
python -m src.main --config configs/paysim.yaml
```

Kết quả được lưu tại:

```text
outputs/results/reproduction_metrics.json
outputs/checkpoints/tssgc_classifier.pt
outputs/checkpoints/dqn_threshold_agent.pt
```

## Pipeline

```text
Raw Transaction Data
        ↓
Train / Validation / Test Split
        ↓
Preprocessing
        ↓
Transaction Graph Construction
        ↓
TSSGC Encoder
        ↓
Classifier Head
        ↓
Fraud Score
        ↓
DQN Threshold Adjustment
        ↓
Fraud / Legitimate Prediction
        ↓
Evaluation
```

## Metric

Project tính:

- AUC-ROC
- AUC-PR
- F1-score
- Precision
- Recall
- Recall@1%
- FPR
- FNR

## Kiểm tra nhanh

```bash
python -m pytest tests
```

Hoặc chỉ compile toàn bộ code:

```bash
python -m compileall src scripts tests
```

## Giới hạn của bản reproduction

Do tác giả chưa public source code, một số chi tiết phải diễn giải kỹ thuật:

1. TSSGC temporal branch dùng time-decay + GRUCell để hiện thực hóa ý tưởng GRU time-aware.
2. NAF trong paper được hiện thực ở mức DQN threshold adjustment rời rạc để chạy ổn định trong offline fraud dataset.
3. Federated Learning chưa bật mặc định trong bản này vì em đang dựng baseline đơn máy trước. Có thể thêm FedAvg ở giai đoạn sau.
4. Với dataset quá lớn, graph builder có `max_neighbors_per_node` để tránh O(N²) bộ nhớ. Đặt `null` nếu muốn exact exhaustive graph trên sample nhỏ.

