import pytest
pytest.importorskip('torch_geometric')

import numpy as np
from src.graph.build_graph import build_transaction_graph


def test_build_transaction_graph():
    cfg = {'dataset': {'time_unit':'hour'}, 'graph': {'time_window_hours':1.0, 'similarity_threshold':0.0, 'max_neighbors_per_node':2, 'add_self_loops':True}}
    x = np.array([[1,0],[0.9,0.1],[0,1]], dtype='float32')
    y = np.array([0,1,0])
    t = np.array([1,1.5,5], dtype='float32')
    g = build_transaction_graph(x, y, t, cfg)
    assert g.x.shape == (3,2)
    assert g.y.shape[0] == 3
    assert g.edge_index.shape[0] == 2
