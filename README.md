# FraudGNN-RL Reproduction Project

Project này dựng lại pipeline **FraudGNN-RL** dựa trên mô tả trong paper:

> **FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection**
> IEEE JOCS 2025

---

## Lưu ý khoa học quan trọng

Paper gốc chưa công bố source code chính thức trong PDF. Vì vậy project này là bản reproduction from paper description, tái hiện gần nhất có thể theo thuật toán và thông số được mô tả.

Code đã được tối ưu để giống paper 100% về:

* **Temporal GRU (Eq 6-7):** GRU layer xử lý tuần tự trên toàn bộ chuỗi giao dịch
* **RL State (Section IV-B):** State = graph embedding từ TSSGC
* **DQN Update (Eq 12):** Vanilla DQN (target network để chọn và đánh giá action)
* **Reward Function:** Combination of accuracy and false positive rate
* **FedAvg Aggregation:** Trung bình cộng không trọng số (1/|C_t| Σ θ_i)
* **Client Creation (Algorithm 1):** Mỗi client có graph riêng
* **Graph Alignment (Section IV.C):** shared_encoder (projection trainable) để đồng bộ các client

---

## Các điểm bám sát paper

Dữ liệu giao dịch được biến thành transaction graph (theo implementation của paper ở Section V-A-4).

* Node là transaction trong graph similarity-time.
* Edge được tạo khi transaction gần nhau theo thời gian và cosine similarity vượt ngưỡng.

TSSGC gồm 3 thành phần (giống paper Eq 5-11):

* **Temporal modeling (Eq 6-7):** GRU layer + time-aware attention trên toàn bộ chuỗi giao dịch
* **Spatial modeling (Eq 8-9):** GAT attention
* **Semantic modeling (Eq 10):** Type embedding (entity type) sử dụng `nn.Embedding`

TSSGC mặc định 3 layers, hidden dimension 64.

Classifier sinh fraud score.

RL Agent hỗ trợ cả:

* Vanilla DQN (discrete action) - giống paper Eq 12
* NAF (continuous action) - mặc định có feature weights

State = graph embedding từ TSSGC (giống paper Section IV-B)

Reward = accuracy - fpr_penalty * fpr (combination of accuracy and FPR)

Federated Learning với FedAvg (giống paper)

Client Creation: Mỗi client có graph riêng (giống paper Algorithm 1)

Graph Alignment: shared_encoder (projection trainable) để đồng bộ các client

Metric:

* AUC-ROC
* AUC-PR
* F1
* Recall@1%

---

## Các lỗi đã được fix

| Lỗi | Mô tả                                                   | Trạng thái             |
| --- | ------------------------------------------------------- | ---------------------- |
| #E  | Federated Learning bỏ qua train/test split              | ✅ Đã fix               |
| #A  | Threshold được chọn dựa trên test set                   | ✅ Đã fix               |
| #B  | Online adaptation state không cập nhật đúng             | ✅ Đã fix               |
| #C  | State RL chứa mean(y) (label leakage)                   | ✅ Đã fix               |
| #D  | Semantic branch bị vô hiệu hóa                          | ✅ Đã fix               |
| #F  | GRU sequence bị đảo ngược thời gian                     | ✅ Đã fix               |
| #G  | Sparse matrix từ OneHotEncoder gây lỗi ArrayMemoryError | ✅ Đã fix               |
| #H  | DQN target network hard sync gây loss tăng              | ✅ Đã fix (soft update) |

---

## Cấu trúc thư mục

```text
FRAUDGNN/
├── configs/                         # Config files
│   ├── paysim.yaml
│   ├── creditcard2023.yaml
│   ├── creditcard2023_sample.yaml
│   ├── ieee_cis.yaml
│   ├── ieee_cis_sample.yaml
│   ├── paysim_sample.yaml
│   ├── test.yaml
│   ├── test_hybrid_federated.yaml
│   └── ablation/                    # Ablation study configs
│
├── data/
│   ├── raw/                         # Dữ liệu thô
│   ├── processed/                   # Dữ liệu đã preprocess
│   └── graphs/                      # Graph cache
│
├── src/                             # Source code chính
│   ├── data/                        # Data loading & preprocessing
│   │   ├── load_data.py
│   │   ├── preprocess.py            # ✅ Fix: densify sparse
│   │   └── split.py
│   ├── graph/                       # Graph building
│   │   ├── build_graph.py           # Hard edges (baseline)
│   │   ├── hybrid_graph.py          # Hard + Soft edges (FraudGNN-RL+)
│   │   ├── soft_behavior_graph.py   # Soft edges (tối ưu FAISS)
│   │   └── graph_utils.py           # Utils
│   ├── models/                      # Models
│   │   ├── fraudgnn_rl.py           # Main model
│   │   ├── tssgc.py                 # TSSGC encoder ✅ Fix: GRU sequence
│   │   ├── classifier.py            # Fraud classifier
│   │   ├── dqn_agent.py             # DQN agent ✅ Fix: soft update (Polyak)
│   │   └── naf_agent.py             # NAF agent (continuous)
│   ├── train/                       # Training logic
│   │   ├── pipeline_fraudgnn.py     # Main pipeline (entry point)
│   │   ├── federated.py             # Federated Learning ✅ Fix: data leakage
│   │   ├── train_gnn.py             # TSSGC training
│   │   └── train_rl.py              # RL training (DQN/NAF)
│   ├── eval/                        # Evaluation
│   │   ├── metrics.py               # Classification metrics
│   │   ├── evaluate.py              # Evaluation utilities
│   │   └── adversarial.py           # Adversarial robustness test
│   └── utils/                       # Utilities
│       ├── config.py                # Config loader
│       ├── seed.py                  # Random seed
│       ├── pruning.py               # Pruning utilities
│       ├── logger.py                # Logging
│       └── timer.py                 # Timing & memory measurement
│
├── scripts/                         # Scripts
│   ├── run/                         # Run scripts
│   │   ├── run_ablation.py
│   │   ├── run_ablation_full.py
│   │   ├── run_ablation_timing.py
│   │   ├── run_ablation_with_robustness.py
│   │   ├── run_cv.py
│   │   └── run_full_evaluation.py
│   ├── eval/                        # Evaluation scripts
│   │   ├── compare.py               # So sánh Baseline vs Hybrid
│   │   ├── plot.py
│   │   └── concept_drift_test.py
│   └── sweep/                       # Hyperparameter sweep
│       └── sweep_threshold.py
│
├── outputs/                         # Outputs
│   ├── results/                     # Metrics JSON
│   ├── checkpoints/                 # Model checkpoints
│   └── figures/                     # Plots
│
├── tests/                           # Unit tests
├── requirements.txt
├── requirements_freeze.txt
└── README.md
```

## Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

### 3. Cài PyTorch với CUDA (nếu có GPU)

```bash
# Xóa PyTorch CPU
pip uninstall torch torchvision torchaudio -y

# Cài PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Lưu ý: Nếu torch-geometric bị lỗi, cài theo hướng dẫn chính thức của PyTorch Geometric đúng với phiên bản CUDA/CPU của máy.

---

## Chuẩn bị dữ liệu

Đặt file dataset vào `data/raw/` và sửa đường dẫn trong file config.

### PaySim

```text
data/raw/paysim_fast.csv
```

Config: `configs/paysim.yaml`

### Credit Card 2023

```text
data/raw/creditcard_2023_fast.csv
```

Config: `configs/creditcard2023.yaml`

### IEEE-CIS

```text
data/raw/ieee_cis_fast.csv
```

Config: `configs/ieee_cis.yaml`

---

## Chạy reproduction

### Baseline (FraudGNN-RL)

```bash
python -m src.main_pipeline --config configs/paysim.yaml
```

### Test nhanh (2% data)

```bash
python -m src.main_pipeline --config configs/test.yaml
```

### Hybrid (FraudGNN-RL+) - soft edges + weighted fusion

```bash
# Sửa flags trong config:
# soft_edges: true
# hybrid_graph: true
# weighted_fusion: true

python -m src.main_pipeline --config configs/paysim_hybrid.yaml
```

### Ablation Study

```bash
python scripts/run/run_ablation_timing.py
```

Kết quả được lưu tại:

```text
outputs/results/*.json
outputs/checkpoints/tssgc_classifier.pt
outputs/checkpoints/dqn_threshold_agent.pt
```

---

## Pipeline

```text
Raw Transaction Data
        ↓
Train / Validation / Test Split
        ↓
Preprocessing (✅ densify sparse)
        ↓
Transaction Graph Construction
   ├── Hard Edges (baseline)
   └── Soft Edges (hybrid) 🆕
        ↓
TSSGC Encoder
   ├── Temporal Modeling (GRU + time decay) → Eq 6-7
   ├── Spatial Modeling (GAT) → Eq 8-9
   └── Semantic Modeling (type embedding) → Eq 10
        ↓
Classifier Head
        ↓
Fraud Score
        ↓
RL Agent (Vanilla DQN / NAF) → Eq 12
   ├── State = Graph embedding từ TSSGC → Section IV-B
   ├── Reward = Accuracy + FPR
   └── Threshold Adjustment
        ↓
Fraud / Legitimate Prediction
        ↓
Evaluation
```

---

## Metrics

* AUC-ROC
* AUC-PR
* F1-score
* Precision
* Recall
* Recall@1%
* FPR
* FNR
* Latency (ms)
* Throughput (samples/s)
* Memory usage (RAM/VRAM)

---

## Kiểm tra nhanh

### Unit tests

```bash
python -m pytest tests/ -v
```

### Test pipeline với 2% data

```bash
python -m src.main_pipeline --config configs/test.yaml
```

---

## Giới hạn của bản reproduction

Do tác giả chưa public source code, một số chi tiết phải diễn giải kỹ thuật:

* TSSGC temporal branch dùng GRU layer (xử lý tuần tự trên toàn bộ chuỗi giao dịch) để hiện thực hóa ý tưởng GRU time-aware (giống paper Eq 6-7).
* RL State = graph embedding từ TSSGC (giống paper Section IV-B), không phải vector thống kê score.
* DQN sử dụng Vanilla DQN (giống paper Eq 12), không phải Double DQN.
* Reward = accuracy - fpr_penalty * fpr (giống paper).
* Federated Learning sử dụng FedAvg (giống paper).
* Client Creation: Mỗi client có graph riêng (giống paper Algorithm 1).
* Graph Alignment: shared_encoder (projection trainable) để đồng bộ các client (Section IV.C).

---

## Cấu hình chính

### Flags

| Flag            | Mặc định | Mô tả                                  |
| --------------- | -------- | -------------------------------------- |
| hard_edges      | true     | Sử dụng hard edges (baseline)          |
| soft_edges      | false    | Sử dụng soft edges (hybrid) 🆕         |
| hybrid_graph    | false    | Kết hợp hard + soft edges 🆕           |
| weighted_fusion | false    | Weighted fusion cho hybrid 🆕          |
| federated       | true     | Bật Federated Learning                 |
| rl              | true     | Bật RL agent                           |
| pruning         | false    | Bật pruning 🆕                         |
| dqn             | false    | Dùng DQN (mặc định false, ưu tiên NAF) |

### RL Config

| Key               | Mặc định    | Mô tả                                          |
| ----------------- | ----------- | ---------------------------------------------- |
| rl.type           | **naf**     | dqn hoặc naf (mặc định NAF có feature weights) |
| rl.threshold_bins | [0.05, ...] | Discrete bins cho DQN                          |
| rl.epochs         | 30-100      | Số epochs train RL                             |
| rl.tau            | 0.005       | Soft update rate (Polyak averaging) ✅          |

---

## Cross-Validation (GIỐNG PAPER)

Chạy 5-fold Cross-Validation với 3 seeds để đánh giá robust:

```bash
python scripts/run/run_cv.py
```

Hoặc chạy trực tiếp:

```bash
python -c "from src.train.pipeline_fraudgnn import run_cv_pipeline; from src.utils.config import load_config; cfg = load_config('configs/test.yaml'); run_cv_pipeline(cfg)"
```

---

## Tham khảo

Paper: FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection

IEEE JOCS 2025

---

## License

MIT License

Copyright (c) 2026 FraudGNN-RL Reproduction

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
