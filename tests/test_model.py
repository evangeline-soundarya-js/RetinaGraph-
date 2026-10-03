import pytest
import torch
from src.models.gat_model import RetinaGAT

def test_model_initialization():
    model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2, edge_dim=3)
    assert model.config["in_channels"] == 4
    assert model.config["hidden_channels"] == 16
    assert model.config["num_classes"] == 2
    assert model.config["heads"] == 2
    assert model.config["edge_dim"] == 3

def test_forward_pass_dimensions():
    model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2, edge_dim=3)
    model.eval()
    
    # 3 nodes, 4 features
    x = torch.rand((3, 4))
    # 2 edges
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    # 2 edges, 3 features
    edge_attr = torch.rand((2, 3))
    
    with torch.no_grad():
        out, attention = model(x, edge_index, edge_attr=edge_attr)
        
    assert out.shape == (1, 2) # [batch_size, num_classes], batch=1 since no batch provided
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    assert "alpha1" in attention
    assert "alpha2" in attention

def test_batching():
    model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2, edge_dim=3)
    model.eval()
    
    # 5 nodes total, across 2 graphs
    x = torch.rand((5, 4))
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)
    edge_attr = torch.rand((3, 3))
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    
    with torch.no_grad():
        out, _ = model(x, edge_index, edge_attr=edge_attr, batch=batch)
        
    assert out.shape == (2, 2) # 2 graphs, 2 classes

def test_empty_graph_handling():
    # If the graph has 1 node and 0 edges, the GAT should still process it via self-loops 
    # automatically added by PyG GATConv (if we didn't disable them).
    model = RetinaGAT(in_channels=4, hidden_channels=16, num_classes=2, heads=2, edge_dim=3)
    model.eval()
    
    x = torch.rand((1, 4))
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 3), dtype=torch.float32)
    
    with torch.no_grad():
        out, _ = model(x, edge_index, edge_attr=edge_attr)
        
    assert out.shape == (1, 2)
