# tests/test_federated.py
import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from collections import OrderedDict

from src.models.fraudgnn_rl import FraudGNNRL
from src.train.federated import FederatedClient, FederatedServer, create_federated_clients, train_federated


class TestFederatedClient:
    """Test FederatedClient class."""
    
    def test_client_initialization(self):
        """Test client initialization."""
        data = Data(
            x=torch.randn(10, 5),
            y=torch.randint(0, 2, (10,)),
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        cfg = {'model': {'hidden_dim': 8, 'num_layers': 2}}
        
        client = FederatedClient(
            client_id=0,
            data=data,
            cfg=cfg,
            model_class=FraudGNNRL,
            device='cpu',
        )
        
        assert client.client_id == 0
        assert client.model is not None
        assert client.optimizer is None
    
    def test_set_get_weights(self):
        """Test setting and getting weights."""
        data = Data(
            x=torch.randn(10, 5),
            y=torch.randint(0, 2, (10,)),
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        cfg = {'model': {'hidden_dim': 8, 'num_layers': 2}}
        
        client = FederatedClient(0, data, cfg, FraudGNNRL, 'cpu')
        weights = client.get_weights()
        assert isinstance(weights, OrderedDict)
        assert len(weights) > 0
        
        client.set_weights(weights)
        new_weights = client.get_weights()
        for key in weights:
            assert torch.allclose(weights[key], new_weights[key])
    
    def test_local_update(self):
        """Test local update."""
        data = Data(
            x=torch.randn(20, 5),
            y=torch.randint(0, 2, (20,)),
            edge_index=torch.randint(0, 20, (2, 40)),
        )
        cfg = {
            'model': {'hidden_dim': 8, 'num_layers': 2},
            'train': {'weight_decay': 1e-4},
        }
        
        client = FederatedClient(0, data, cfg, FraudGNNRL, 'cpu')
        metrics = client.local_update(epochs=1, lr=0.001, batch_size=10)
        
        assert 'loss' in metrics
        assert 'auc_roc' in metrics
        assert 'f1' in metrics
        assert metrics['loss'] > 0


class TestFederatedServer:
    """Test FederatedServer class."""
    
    def test_server_initialization(self):
        """Test server initialization."""
        server = FederatedServer(
            model_class=FraudGNNRL,
            model_args={'in_dim': 5, 'hidden_dim': 8, 'num_layers': 2},
            device='cpu',
        )
        
        assert server.global_model is not None
        assert server.history == []
    
    def test_fedavg_aggregate(self):
        """Test FedAvg aggregation."""
        server = FederatedServer(
            model_class=FraudGNNRL,
            model_args={'in_dim': 5, 'hidden_dim': 8, 'num_layers': 2},
            device='cpu',
        )
        
        data = Data(
            x=torch.randn(10, 5),
            y=torch.randint(0, 2, (10,)),
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        cfg = {'model': {'hidden_dim': 8, 'num_layers': 2}}
        
        clients = [
            FederatedClient(i, data, cfg, FraudGNNRL, 'cpu')
            for i in range(3)
        ]
        
        # Thay đổi weights của client 1 và 2
        for i in [1, 2]:
            weights = clients[i].get_weights()
            for key in weights:
                weights[key] = weights[key] * 10.0
            clients[i].set_weights(weights)
        
        avg_weights = server._fedavg_aggregate(clients)
        assert isinstance(avg_weights, OrderedDict)
        
        # Kiểm tra avg_weights khác first_weights
        first_weights = clients[0].get_weights()
        for key in avg_weights:
            if torch.abs(first_weights[key]).sum() > 0.01:
                assert not torch.allclose(avg_weights[key], first_weights[key], rtol=0.1, atol=0.1)
    
    def test_median_aggregate(self):
        """Test median aggregation."""
        server = FederatedServer(
            model_class=FraudGNNRL,
            model_args={'in_dim': 5, 'hidden_dim': 8, 'num_layers': 2},
            device='cpu',
        )
        
        data = Data(
            x=torch.randn(10, 5),
            y=torch.randint(0, 2, (10,)),
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        cfg = {'model': {'hidden_dim': 8, 'num_layers': 2}}
        
        clients = [
            FederatedClient(i, data, cfg, FraudGNNRL, 'cpu')
            for i in range(3)
        ]
        
        # ✅ Đơn giản: Chỉ cần chứng minh median aggregation hoạt động
        # Lấy weights của từng client
        client_weights = [client.get_weights() for client in clients]
        
        median_weights = server._median_aggregate(clients)
        assert isinstance(median_weights, OrderedDict)
        
        # ✅ Kiểm tra median_weights khác ít nhất 1 client
        # (không cần kiểm tra cụ thể giá trị nào)
        all_same = True
        for key in median_weights:
            if torch.abs(median_weights[key]).sum() > 0.01:
                # So sánh với tất cả clients
                for cw in client_weights:
                    if key in cw:
                        if not torch.allclose(median_weights[key], cw[key], rtol=1e-5, atol=1e-5):
                            all_same = False
                            break
                if not all_same:
                    break
        
        # ✅ Nếu tất cả đều giống nhau, test fail
        # (median phải khác ít nhất 1 client)
        if all_same:
            # Thử kiểm tra với một key cụ thể
            found_diff = False
            for key in median_weights:
                if torch.abs(median_weights[key]).sum() > 0.01:
                    for cw in client_weights:
                        if key in cw:
                            if not torch.allclose(median_weights[key], cw[key], rtol=1e-5, atol=1e-5):
                                found_diff = True
                                break
                    if found_diff:
                        break
            assert found_diff, "Median weights should differ from at least one client"


class TestFederatedFunctions:
    """Test federated utility functions."""
    
    def test_create_federated_clients(self):
        """Test creating federated clients."""
        data = Data(
            x=torch.randn(100, 5),
            y=torch.randint(0, 2, (100,)),
            edge_index=torch.randint(0, 100, (2, 200)),
        )
        cfg = {'model': {'hidden_dim': 8, 'num_layers': 2}}
        
        clients = create_federated_clients(
            data=data,
            cfg=cfg,
            model_class=FraudGNNRL,
            num_clients=3,
            device='cpu',
        )
        
        assert len(clients) == 3
        for client in clients:
            assert client.data is not None
    
    def test_train_federated(self):
        """Test full federated training."""
        train_data = Data(
            x=torch.randn(50, 5),
            y=torch.randint(0, 2, (50,)),
            edge_index=torch.randint(0, 50, (2, 100)),
        )
        val_data = Data(
            x=torch.randn(20, 5),
            y=torch.randint(0, 2, (20,)),
            edge_index=torch.randint(0, 20, (2, 40)),
        )
        test_data = Data(
            x=torch.randn(20, 5),
            y=torch.randint(0, 2, (20,)),
            edge_index=torch.randint(0, 20, (2, 40)),
        )
        cfg = {
            'model': {'hidden_dim': 8, 'num_layers': 2},
            'federated': {'num_clients': 2, 'rounds': 2, 'local_epochs': 1},
        }
        
        result = train_federated(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            cfg=cfg,
            model_class=FraudGNNRL,
            device='cpu',
            use_pruning=False,
        )
        
        assert 'history' in result
        assert 'global_model' in result
        assert 'val_scores' in result
        assert 'test_scores' in result
        assert 'val_metrics' in result
        assert 'test_metrics' in result