# FraudGNN-RL — Reproduction

Tái hiện **FraudGNN-RL: A Graph Neural Network With Reinforcement Learning for Adaptive Financial Fraud Detection**
(Cui et al., IEEE Open J. Comput. Soc., 2025). Paper không công bố code nên đây là bản dựng lại **từ mô tả trong paper**:
phần paper nêu rõ được làm đúng nguyên văn; phần không rõ được chọn phương án hợp lý nhất và đánh dấu `[INFERRED]` ngay trong code.

> **Quy tắc dữ liệu:** project không bao giờ tự sinh/giả lập dữ liệu. Thiếu file => báo lỗi rõ ràng.
> `verify_all.py --unit` dùng tensor ngẫu nhiên nhỏ chỉ để test đúng/sai của code, không tạo ra metric nào.

## 1. Paper nêu rõ → cài đặt đúng nguyên văn

| Thành phần | Paper | Code |
|---|---|---|
| Graph | node = transaction; nối nếu trong **1 giờ** và **cosine ≥ 0.9** (V-A-4) | `graph/graph_utils.py` (tìm kiếm chính xác, đã kiểm chứng với brute-force) |
| TSSGC | Eq. 5–11: GRU + attention thời gian (Eq. 6–7), GAT (Eq. 8–9), Semantic embedding (Eq. 10) | `models/tssgc.py` |
| Kiến trúc | 3 lớp, 64 hidden, ReLU, BatchNorm sau mỗi lớp | `models/tssgc.py` |
| RL | State = graph embedding; Action = threshold + feature weights; Reward kết hợp accuracy & FPR; NAF cho action liên tục; loss Eq. 12 (target network) | `models/naf_agent.py`, `reward.py` |
| RL net | 2 hidden × 128, Adam lr 1e-3 | `naf_agent.py` |
| FL | Algorithm 1, FedAvg **trung bình đều** 1/\|C_t\|, mỗi client một đồ thị cục bộ | `train/federated.py` |
| Huấn luyện | Adam lr 1e-3, 100 epoch, early stopping patience 10, seed 42 | `train/train_gnn.py`, `configs/*.yaml` |
| Preprocess | impute mean/mode, min-max, one-hot, hour/day-of-week/Δt; IEEE-CIS merge identity | `data/preprocess.py`, `load_data.py` |
| Metrics | AUC-ROC, AUC-PR, F1, Recall@1% | `eval/metrics.py` |
| Ablation | w/o GNN, w/o RL, w/o FL, w/o GNN&RL, w/o GNN&FL, w/o RL&FL | `utils/model_definitions.py`, `scripts/run/run_ablation.py` |
| 5-fold CV | V-A-4 | `scripts/run/run_cv.py`, `--cv` |

## 2. Paper không rõ → lựa chọn của mình (`[INFERRED]`)

| Vấn đề | Lựa chọn | Lý do |
|---|---|---|
| Công thức reward | `F1 + 0.25·Recall − 1.0·FPR` | accuracy thuần bị lớp normal chi phối khi fraud ≈ 0.1% |
| Feature weights của RL | `w = D·a/Σa` (mean = 1), nhân vào embedding trước classifier; RL tính lại điểm số trong môi trường | bản cũ tính weights nhưng **không** ảnh hưởng điểm test |
| RL train trên tập nào | VAL; áp lên TEST không cần nhãn, không update online | test độc lập; online learning là *future work* của paper |
| Threshold cuối | trung bình threshold policy greedy trên val (`rl.inference: mean`) hoặc theo batch (`per_batch`) | paper không nói |
| Graph alignment (FL) | schema đặc trưng chung + projection `Linear(in→64)` được FedAvg | phương án B đã chốt; áp dụng cả non-FL để ablation công bằng |
| Chia client | 3 client, shard liên tục theo thời gian | giữ cấu trúc thời gian cục bộ |
| FL budget | 50 round × 2 local epoch (= 100 epoch tương đương), patience 10 round | khớp "100 epochs, patience 10" |
| Split | train 0.64 / val 0.16 / test 0.20; temporal (PaySim, IEEE), stratified (CC2023) | "80-20 random + preserving temporal order" tự mâu thuẫn |
| Batch size 64 | `batch_mode: subgraph` (chunk liên tục theo thời gian) hoặc `full` | batch 64 node không khả thi cho TSSGC 3-hop |
| Class imbalance | BCE với `pos_weight = n_neg/n_pos` (cắt ở 100) | paper nói có xử lý nhưng không nêu cách |
| Cạnh | vô hướng, không self-loop; Δt = \|t_i − t_j\| | Eq. 6 dùng "edges connected to node i" |
| Baselines | siêu tham số tự chọn (xem đầu `models/baselines.py`); GraphSemi đơn giản hoá thành GraphSAGE | paper chỉ nêu tên |

### `[DEVIATION]` — cap kỹ thuật (ghi vào kết quả `deviations_from_paper`)
- `graph.max_neighbors_per_node` (PaySim/IEEE: 32, CC2023: 16) và `model.temporal_max_seq_len`: paper không giới hạn, nhưng 6.3M giao dịch sẽ nổ bộ nhớ.
  Đặt `null` để chạy đúng paper nếu máy đủ RAM/VRAM.

## 3. Cảnh báo về chính paper (nên đọc trước khi so số)
1. **CC2023**: bản Kaggle `nelgiriyewithana/credit-card-fraud-dataset-2023`, theo hiểu biết của mình, *không có cột thời gian* và có tỉ lệ fraud ≈ 50% — khác với mô tả "temporal patterns / 0.172%" trong paper. Hãy kiểm tra file của bạn; nếu không có time, graph chỉ dùng cosine (config đã xử lý: `time_col: null`).
2. **Table 2 nội tại không nhất quán**: AUC-PR ≈ 0.65 nhưng F1 ≈ 0.92 và Recall@1% ≈ 97% khó cùng tồn tại trên một tập test. Đừng coi các con số này là mục tiêu phải khớp; `scripts/eval/compare.py` chỉ in chênh lệch để tham chiếu.
3. Paper dùng PyTorch 1.8 / PyG 2.0.1; code này dùng `torch>=2.1`, `torch-geometric>=2.4`.

## 4. Sửa so với bản repo cũ
- Temporal features (hour/dow/Δt) trước đây bị loại khỏi X → nay đưa vào.
- Impute categorical: `astype(str)` biến NaN thành lớp `"nan"` → nay giữ NaN để mode-imputation thực sự chạy.
- `node_type` IEEE-CIS dùng `cat.codes` theo từng split (mã lệch) → nay mapping cố định từ config.
- Graph từng dùng kNN k=64 xấp xỉ + vòng lặp Python → nay chính xác, vector hoá.
- Temporal aggregator vector hoá (không còn vòng lặp theo node), có test đối chiếu GRU tuần tự.
- NAF: `mu` ở không gian logit còn action lưu ở không gian đã squash → nay cùng không gian [0,1].
- Feature weights của RL nay thực sự tác động lên điểm test.

## 5. Cài đặt & chạy
```bash
pip install -r requirements.txt          # PyG: theo hướng dẫn chính thức nếu lỗi
python verify_all.py --unit              # kiểm tra toán/shape (không cần dữ liệu)
python audit_runtime.py --config configs/paysim.yaml
```
Đặt dữ liệu vào `data/raw/` (đường dẫn trong `configs/*.yaml`):
`paysim_fast.csv`, `creditcard_2023.csv`, `train_transaction.csv` + `train_identity.csv`.

```bash
python verify_all.py --config configs/paysim_sample.yaml        # data contract trên dữ liệu thật
python main.py --config configs/paysim_sample.yaml              # chạy mẫu nhanh (KHÔNG phải kết quả paper)
python main.py --config configs/paysim.yaml                     # bản đầy đủ
python main.py --config configs/paysim.yaml --cv                # 5-fold CV
python main.py --config configs/paysim.yaml --set flags.rl=false graph.max_neighbors_per_node=16

python scripts/run/run_ablation.py --config configs/paysim.yaml
python scripts/run/run_full_evaluation.py --config configs/paysim.yaml --imbalance
python scripts/eval/compare.py --results outputs/results/paysim_paper_repro_full_evaluation.json --dataset paysim
python scripts/run/run_ablation_with_robustness.py --config configs/paysim.yaml   # adversarial + concept drift
```
Kaggle: `!python main.py --config configs/creditcard2023_sample.yaml` (P100/T4 16GB: dùng `*_sample` hoặc giữ cap láng giềng).

## 6. Model của bạn (extension, mặc định TẮT)
| Model | flags |
|---|---|
| Baseline (= paper) | `hard_edges: true` |
| Augmented Graph | `hard_edges: true, soft_edges: true, pruning: true` |

Cùng seed, cùng split, chỉ đổi `flags` → so sánh bằng `scripts/eval/compare.py --a ... --b ...`. Chi tiết: `src/extensions/EXTENSIONS.md`.

## 7. Cấu trúc
Xem `note.txt` cho luồng xử lý. File ngoài cây gốc bạn yêu cầu: `src/models/baselines.py`, `scripts/utils/common.py`.
