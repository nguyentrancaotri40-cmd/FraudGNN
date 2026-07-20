import pytest
pytest.importorskip('torch_geometric')

import torch
from torch_geometric.data import Data
from src.models.fraudgnn_rl import FraudGNNRL


def test_model_forward():
    data = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor([[0,1,2,3,0,1,2,3],[1,2,3,0,0,1,2,3]], dtype=torch.long),
        edge_time_delta=torch.zeros(8),
        node_type=torch.zeros(4, dtype=torch.long),
        y=torch.tensor([0,1,0,1]),
    )
    model = FraudGNNRL(in_dim=5, hidden_dim=8, num_layers=2)
    logits = model(data)
    assert logits.shape == (4,)
