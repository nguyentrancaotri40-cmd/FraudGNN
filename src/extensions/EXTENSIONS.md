# FraudGNN-RL+ Extensions (Augmented Graph)

> **LƯU Ý**: mọi thứ trong thư mục này là **CẢI TIẾN NGOÀI PAPER** (Cui et al., IEEE OJCS 2025).
> Mặc định TẮT; bản reproduction 100% paper chạy với `flags: {hard_edges: true, soft_edges: false, pruning: false}`.

| Model | flags |
|---|---|
| Baseline (= paper) | `hard_edges: true` |
| Augmented Graph (model của bạn) | `hard_edges: true, soft_edges: true, pruning: true` |
| Ablation: Soft only | `hard_edges: false, soft_edges: true` |
| Ablation: Pruning only | `hard_edges: true, pruning: true` |

## Soft behavioral edges (`soft_behavior_graph.py`)
- Cạnh *mềm* theo độ tương đồng hành vi: cosine >= `soft_graph.similarity_threshold` (mặc định 0.85,
  thấp hơn ngưỡng 0.9 của hard edge) trên toàn bộ feature hoặc tập `feature_indices`.
- Giữ tối đa `max_neighbors_per_node` láng giềng/node (top-k theo cosine).
- **Causal**: chỉ nối từ giao dịch TRƯỚC -> SAU (tránh nhìn tương lai). Khi dataset không có thời gian
  (CC2023) "trước/sau" theo thứ tự hàng, nên khuyến nghị dùng `causal: false`.
- Hợp nhất với hard edges (loại trùng, giữ delta nhỏ nhất) trước khi vào TSSGC.
- `edge_weight` (cosine) được trả về để phân tích; TSSGC (GAT) học trọng số attention riêng.

## Pruning (`pruning.py`)
- L1 magnitude pruning theo tầng, lịch tăng dần `initial_amount -> final_amount`
  giữa `start_epoch` và `end_epoch`; zero hoá trực tiếp (không dùng mask hook) nên tương thích FedAvg.
- Ghi `pruning_ratio` thực tế vào kết quả để báo cáo trade-off recall/FPR/precision & latency.

## Quy tắc so sánh công bằng
Baseline và Augmented phải dùng CÙNG seed, CÙNG split, CÙNG hyper-parameter; chỉ khác `flags`.
