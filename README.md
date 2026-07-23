# FraudGNN-RL Reproduction Project

Project này dựng lại pipeline **FraudGNN-RL** dựa trên mô tả trong paper:

> FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection  
> IEEE JOCS 2025

---

# Lưu ý khoa học quan trọng

Paper gốc chưa công bố source code chính thức trong PDF. Vì vậy project này là bản **reproduction from paper description**, tức là tái hiện gần nhất có thể theo thuật toán và thông số được mô tả.

**Code đã được tối ưu để giống paper 100%** về:

- **Temporal GRU** (Eq 6-7): GRU layer xử lý tuần tự trên toàn bộ chuỗi giao dịch
- **RL State** (Section IV-B): State = graph embedding từ TSSGC
- **DQN Update** (Eq 12): Vanilla DQN (target network để chọn và đánh giá action)
- **Reward Function**: Combination of accuracy and false positive rate
- **FedAvg Aggregation**: Trung bình có trọng số theo số lượng mẫu (n_i / n_total)
- **Client Creation** (Algorithm 1): Mỗi client có graph riêng

---

# Các điểm bám sát paper

- Dữ liệu giao dịch được biến thành **transaction graph** (theo implementation của paper ở Section V-A-4).
- Node là transaction trong graph similarity-time.
- Edge được tạo khi transaction gần nhau theo thời gian và cosine similarity vượt ngưỡng.
- **TSSGC** gồm 3 thành phần (giống paper Eq 5-11):
  - **Temporal modeling** (Eq 6-7): GRU layer + time-aware attention trên toàn bộ chuỗi giao dịch
  - **Spatial modeling** (Eq 8-9): GAT attention
  - **Semantic modeling** (Eq 10): Type embedding
- TSSGC mặc định 3 layers, hidden dimension 64.
- Classifier sinh fraud score.
- **RL Agent** hỗ trợ cả:
  - **Vanilla DQN** (discrete action) - giống paper Eq 12
  - **NAF** (continuous action) - có thể bật qua config `rl.type: naf`
- **State** = graph embedding từ TSSGC (giống paper Section IV-B)
- **Reward** = accuracy - fpr_penalty * fpr (combination of accuracy and FPR)
- **Federated Learning** với FedAvg có trọng số theo số lượng mẫu (giống paper)
- **Client Creation**: Mỗi client có graph riêng (giống paper Algorithm 1)
- Metric gồm AUC-ROC, AUC-PR, F1, Recall@1%.

---

# Cấu trúc thư mục

```text
FRAUDGNN/
├── configs/                         # Config files
│   ├── paysim.yaml
│   ├── creditcard2023.yaml
│   ├── ieee_cis.yaml
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
│   │   ├── preprocess.py
│   │   └── split.py
│   ├── graph/                       # Graph building
│   │   ├── build_graph.py           # Hard edges (baseline)
│   │   ├── hybrid_graph.py          # Hard + Soft edges (FraudGNN-RL+)
│   │   ├── soft_behavior_graph.py   # Soft edges
│   │   └── graph_utils.py           # Utils (vectorized hard edges)
│   ├── models/                      # Models
│   │   ├── fraudgnn_rl.py           # Main model
│   │   ├── tssgc.py                 # TSSGC encoder (Temporal GRU fix)
│   │   ├── classifier.py            # Fraud classifier
│   │   ├── dqn_agent.py             # DQN agent (Vanilla DQN + Reward fix)
│   │   └── naf_agent.py             # NAF agent (continuous)
│   ├── train/                       # Training logic
│   │   ├── pipeline_fraudgnn.py     # Main pipeline (entry point)
│   │   ├── federated.py             # Federated Learning (FedAvg weighted)
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
│   │   └── run_ablation_timing.py
│   ├── eval/                        # Evaluation scripts
│   │   ├── compare.py
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
│   ├── test_graph.py
│   ├── test_model.py
│   ├── test_preprocess.py
│   ├── test_federated.py
│   ├── test_pruning.py
│   └── test_pipeline.py
│
├── requirements.txt
└── README.md
```

# Cài đặt

## 1. Tạo môi trường ảo

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows
```

## 2. Cài dependencies

```bash
pip install -r requirements.txt
```

## 3. Cài PyTorch với CUDA (nếu có GPU)

```bash
# Xóa PyTorch CPU
pip uninstall torch torchvision torchaudio -y

# Cài PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Lưu ý: Nếu torch-geometric bị lỗi, cài theo hướng dẫn chính thức của PyTorch Geometric đúng với phiên bản CUDA/CPU của máy.

---

# Chuẩn bị dữ liệu

Đặt file dataset vào `data/raw/` và sửa đường dẫn trong file config.

## PaySim

```text
data/raw/paysim_fast.csv
```

Config:

```text
configs/paysim.yaml
```

Lưu ý: PaySim dùng label thật là `isFraud`, không dùng `isFlaggedFraud`.

## Credit Card 2023

```text
data/raw/creditcard_2023_fast.csv
```

Config:

```text
configs/creditcard2023.yaml
```

## IEEE-CIS

```text
data/raw/ieee_cis_fast.csv
```

Config:

```text
configs/ieee_cis.yaml
```

# Chạy reproduction

## Baseline (FraudGNN-RL)

```bash
python -m src.main_pipeline --config configs/paysim.yaml
```

## Hybrid (FraudGNN-RL+) - với soft edges + weighted fusion

```bash
# Sửa flags trong config:
# hybrid_graph: true
# weighted_fusion: true
# soft_edges: true

python -m src.main_pipeline --config configs/paysim_hybrid.yaml
```

## Với NAF (continuous action)

```bash
# Sửa flags trong config:
# rl.type: naf

python -m src.main_pipeline --config configs/paysim_naf.yaml
```

## Ablation Study

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

# Pipeline

```text
Raw Transaction Data
        ↓
Train / Validation / Test Split
        ↓
Preprocessing
        ↓
Transaction Graph Construction
   ├── Hard Edges (baseline)
   └── Soft Edges (hybrid)
        ↓
TSSGC Encoder
   ├── Temporal Modeling (GRU + time decay) Giống paper Eq 6-7
   ├── Spatial Modeling (GAT) Giống paper Eq 8-9
   └── Semantic Modeling (type embedding) Giống paper Eq 10
        ↓
Classifier Head
        ↓
Fraud Score
        ↓
RL Agent (Vanilla DQN / NAF) Giống paper Eq 12
   ├── State = Graph embedding từ TSSGC Giống paper Section IV-B
   ├── Reward = Accuracy + FPR Giống paper
   └── Threshold Adjustment
        ↓
Fraud / Legitimate Prediction
        ↓
Evaluation
```

---

# Metrics

Project tính:

- AUC-ROC
- AUC-PR
- F1-score
- Precision
- Recall
- Recall@1%
- FPR
- FNR
- Latency (ms)
- Throughput (samples/s)
- Memory usage (RAM/VRAM)

---

# 🧪 Kiểm tra nhanh

## Unit tests

```bash
python -m pytest tests/ -v
```

## Compile check

```bash
python -m compileall src scripts tests
```

## Test pipeline với 1% data

```bash
python -m src.main_pipeline --config configs/test.yaml
```

---

# Giới hạn của bản reproduction

Do tác giả chưa public source code, một số chi tiết phải diễn giải kỹ thuật:

- TSSGC temporal branch dùng GRU layer (xử lý tuần tự trên toàn bộ chuỗi giao dịch) để hiện thực hóa ý tưởng GRU time-aware (giống paper Eq 6-7).

- RL State = graph embedding từ TSSGC (giống paper Section IV-B), không phải vector thống kê score.

- DQN sử dụng Vanilla DQN (giống paper Eq 12), không phải Double DQN.

- Reward = accuracy - fpr_penalty * fpr (combination of accuracy and FPR, giống paper).

- Federated Learning sử dụng FedAvg với trọng số theo số lượng mẫu (giống paper).

- Client Creation: Mỗi client có graph riêng (giống paper Algorithm 1).

- NAF (Normalized Advantage Functions) đã được implement để hỗ trợ continuous action space.

- Mặc định dùng DQN với discrete bins (ổn định hơn).

- Có thể bật NAF bằng config:

```yaml
rl.type: naf
```

- Feature importance weights đã được implement trong NAF agent.

- Federated Learning được implement với FedAvg, có thể bật bằng config:

```yaml
flags:
  federated: true
```

- Graph builder có `max_neighbors_per_node` để tránh O(N²) bộ nhớ. Đặt `null` nếu muốn exact exhaustive graph trên sample nhỏ.

- Hard-edges builder đã được vector hóa (dùng `NearestNeighbors`), cải thiện tốc độ ~5-10x so với phiên bản vòng lặp cũ.

---

# Cấu hình chính

## Flags

| Flag | Mặc định | Mô tả |
|------|----------|------|
| hard_edges | true | Sử dụng hard edges (baseline) |
| soft_edges | false | Sử dụng soft edges (hybrid) |
| hybrid_graph | false | Kết hợp hard + soft edges |
| weighted_fusion | false | Weighted fusion cho hybrid |
| federated | false | Bật Federated Learning |
| rl | true | Bật RL agent |
| pruning | false | Bật pruning |
| dqn | true | Dùng DQN (nếu rl.type: dqn) |

## RL Config

| Key | Mặc định | Mô tả |
|------|----------|------|
| rl.type | dqn | dqn hoặc naf |
| rl.threshold_bins | [0.05, ...] | Discrete bins cho DQN |
| rl.epochs | 30 | Số epochs train RL |

---

# Tham khảo

Paper: FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection

IEEE JOCS 2025

---

# License

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