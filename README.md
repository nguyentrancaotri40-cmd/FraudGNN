<h1 style="color: #2c974b;"> FraudGNN-RL Reproduction Project</h1>

Project này dựng lại pipeline **FraudGNN-RL** dựa trên mô tả trong paper:

> FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection  
> IEEE JOCS 2025

---


<h2 style="color: #2c974b;"> Lưu ý khoa học quan trọng</h2>

Paper gốc chưa công bố source code chính thức trong PDF. Vì vậy project này là bản **reproduction from paper description**, tức là tái hiện gần nhất có thể theo thuật toán và thông số được mô tả, không thể cam kết giống 100% với mã nguồn nội bộ của tác giả.

<h3 style="color: #2c974b;">Các điểm bám sát paper:</h3>

- Dữ liệu giao dịch được biến thành **transaction graph** (theo implementation của paper ở Section V-A-4).
- Node là transaction trong graph similarity-time.
- Edge được tạo khi transaction gần nhau theo thời gian và cosine similarity vượt ngưỡng.
- **TSSGC** gồm 3 thành phần:
  - Temporal modeling (GRU + time-aware attention)
  - Spatial modeling (GAT attention)
  - Semantic modeling (type embedding)
- TSSGC mặc định 3 layers, hidden dimension 64.
- Classifier sinh fraud score.
- **RL Agent** hỗ trợ cả:
  - **DQN** (discrete action) - mặc định, ổn định hơn
  - **NAF** (continuous action) - có thể bật qua config `rl.type: naf`
- **Feature importance weights** đã được implement trong NAF agent.
- Metric gồm AUC-ROC, AUC-PR, F1, Recall@1%.

---


<h2 style="color: #2c974b;"> Cấu trúc thư mục</h2>

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
│   │   ├── tssgc.py                 # TSSGC encoder
│   │   ├── classifier.py            # Fraud classifier
│   │   ├── dqn_agent.py             # DQN agent (discrete)
│   │   └── naf_agent.py             # NAF agent (continuous)
│   ├── train/                       # Training logic
│   │   ├── pipeline_fraudgnn.py     # Main pipeline (entry point)
│   │   ├── federated.py             # Federated Learning (FedAvg)
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

<h2 style="color: #2c974b;"> Cài đặt</h2><h3>1. Tạo môi trường ảo</h3>
bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

<h3>2. Cài dependencies</h3>
bash
pip install -r requirements.txt

<h3>3. Cài PyTorch với CUDA (nếu có GPU)</h3>
bash
# Xóa PyTorch CPU
pip uninstall torch torchvision torchaudio -y

# Cài PyTorch với CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Lưu ý: Nếu torch-geometric bị lỗi, cài theo hướng dẫn chính thức của PyTorch Geometric đúng với phiên bản CUDA/CPU của máy.

<h2 style="color: #2c974b;"> Chuẩn bị dữ liệu</h2>
Đặt file dataset vào data/raw/ và sửa đường dẫn trong file config.

<h3>PaySim</h3>
text
data/raw/paysim_fast.csv
Config: configs/paysim.yaml

Lưu ý: PaySim dùng label thật là isFraud, không dùng isFlaggedFraud.

<h3>Credit Card 2023</h3>
text
data/raw/creditcard_2023_fast.csv
Config: configs/creditcard2023.yaml

<h3>IEEE-CIS</h3>
text
data/raw/ieee_cis_fast.csv
Config: configs/ieee_cis.yaml

<h2 style="color: #2c974b;"> Chạy reproduction</h2><h3>Baseline (FraudGNN-RL)</h3>
bash
python -m src.main_pipeline --config configs/paysim.yaml

<h3>Hybrid (FraudGNN-RL+) - với soft edges + weighted fusion</h3>
bash
# Sửa flags trong config:
# hybrid_graph: true
# weighted_fusion: true
# soft_edges: true
python -m src.main_pipeline --config configs/paysim_hybrid.yaml

<h3>Với NAF (continuous action)</h3>
bash
# Sửa flags trong config:
# rl.type: naf
python -m src.main_pipeline --config configs/paysim_naf.yaml

<h3>Ablation Study</h3>
bash
python scripts/run/run_ablation_timing.py

<h3>Kết quả được lưu tại:</h3>
text
outputs/results/*.json
outputs/checkpoints/tssgc_classifier.pt
outputs/checkpoints/dqn_threshold_agent.pt

<h2 style="color: #2c974b;"> Pipeline</h2>
text
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
   ├── Temporal Modeling (GRU + time decay)
   ├── Spatial Modeling (GAT)
   └── Semantic Modeling (type embedding)
        ↓
Classifier Head
        ↓
Fraud Score
        ↓
RL Agent (DQN / NAF)
   ├── Threshold Adjustment
   └── Feature Importance Weights (NAF)
        ↓
Fraud / Legitimate Prediction
        ↓
Evaluation

<h2 style="color: #2c974b;"> Metrics</h2>
Project tính:
AUC-ROC
AUC-PR
F1-score
Precision
Recall
Recall@1%
FPR
FNR
Latency (ms)
Throughput (samples/s)
Memory usage (RAM/VRAM)

<h2 style="color: #2c974b;"> Kiểm tra nhanh</h2><h3>Unit tests</h3>
bash
python -m pytest tests/ -v

<h3>Compile check</h3>
bash
python -m compileall src scripts tests

<h3>Test pipeline với 1% data</h3>
bash
python -m src.main_pipeline --config configs/test.yaml

<h2 style="color: #2c974b;"> Giới hạn của bản reproduction</h2>
Do tác giả chưa public source code, một số chi tiết phải diễn giải kỹ thuật:

1. TSSGC temporal branch dùng time-decay + GRUCell để hiện thực hóa ý tưởng GRU time-aware.
2. NAF (Normalized Advantage Functions) đã được implement để hỗ trợ continuous action space (threshold ∈ [0, 1]).
   + Mặc định dùng DQN với discrete bins (ổn định hơn)
   + Có thể bật NAF bằng config: rl.type: naf
   + Feature importance weights đã được implement trong NAF agent.
   + RL agent có thể điều chỉnh cả threshold và feature weights.
3. Federated Learning được implement với FedAvg, nhưng mặc định tắt (do chạy baseline đơn máy). Có thể bật bằng config:
yaml
flags:
  federated: true
4. Graph builder có max_neighbors_per_node để tránh O(N²) bộ nhớ. Đặt null nếu muốn exact exhaustive graph trên sample nhỏ.
5. Hard-edges builder đã được vector hóa (dùng NearestNeighbors), cải thiện tốc độ ~5-10x so với phiên bản vòng lặp cũ.

<h2 style="color: #2c974b;"> Cấu hình chính</h2><h3>Flags</h3>
Flag     	 Mặc định	Mô tả
hard_edges	 true	        Sử dụng hard edges (baseline)
soft_edges	 false	        Sử dụng soft edges (hybrid)
hybrid_graph	 false	        Kết hợp hard + soft edges
weighted_fusion  false	        Weighted fusion cho hybrid
federated	 false	        Bật Federated Learning
rl	         true	        Bật RL agent
pruning	         false	        Bật pruning
dqn	         true	        Dùng DQN (nếu rl.type: dqn)

<h3>RL Config</h3>
Key	            Mặc định	     Mô tả
rl.type	            dqn	             dqn hoặc naf
rl.threshold_bins   [0.05, ...]	     Discrete bins cho DQN
rl.epochs	    30	             Số epochs train RL

<h2 style="color: #2c974b;"> Tham khảo</h2>
Paper: FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection
IEEE JOCS 2025

<h2 style="color: #2c974b;"> License</h2>
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