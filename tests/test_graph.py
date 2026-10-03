import pytest
import numpy as np
import torch
from src.graph.builder import GraphBuilder
from torch_geometric.data import Data

@pytest.fixture
def builder():
    return GraphBuilder(k_neighbors=2)

def test_normal_graph_construction(builder):
    # 3 nodes, k=2, each should have 2 outgoing edges -> 6 edges total
    keypoints = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.float32)
    node_features = np.array([[0.0, 0.0, 100, 1],
                              [0.0, 0.5, 120, 1],
                              [0.5, 0.0, 110, 1]], dtype=np.float32)
                              
    data = builder.build_graph(keypoints, node_features)
    
    assert isinstance(data, Data)
    # Shapes
    assert data.x.shape == (3, 4)
    assert data.edge_index.shape == (2, 6)
    assert data.edge_attr.shape == (6, 3)
    
    # Stats
    assert data.stats["num_nodes"] == 3
    assert data.stats["num_edges"] == 6
    assert data.stats["avg_degree"] == 2.0
    assert data.stats["connected_components"] == 1

def test_fewer_than_k_nodes(builder):
    # 2 nodes, k=2 -> can only have 1 neighbor each -> 2 edges total
    keypoints = np.array([[0, 0], [0, 1]], dtype=np.float32)
    node_features = np.array([[0,0,0,0], [0,0,0,0]], dtype=np.float32)
    
    data = builder.build_graph(keypoints, node_features)
    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, 3)
    assert data.stats["num_nodes"] == 2
    assert data.stats["num_edges"] == 2
    
def test_one_node_graph(builder):
    keypoints = np.array([[0, 0]], dtype=np.float32)
    node_features = np.array([[0,0,0,0]], dtype=np.float32)
    
    data = builder.build_graph(keypoints, node_features)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 3)
    assert data.stats["num_nodes"] == 1
    assert data.stats["num_edges"] == 0
    assert data.stats["isolated_nodes"] == 1

def test_zero_node_graph(builder):
    keypoints = np.empty((0, 2), dtype=np.float32)
    node_features = np.empty((0, 4), dtype=np.float32)
    
    data = builder.build_graph(keypoints, node_features)
    assert data.x.shape == (0, 4)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 3)
    assert data.stats["num_nodes"] == 0
    assert data.stats["num_edges"] == 0

def test_duplicate_coordinates(builder):
    # Two identical points
    keypoints = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.float32)
    node_features = np.zeros((3, 4), dtype=np.float32)
    
    data = builder.build_graph(keypoints, node_features)
    assert data.edge_index.shape == (2, 6) # Each point connects to the other two
    # Ensure no self loops
    for i in range(data.edge_index.shape[1]):
        assert data.edge_index[0, i] != data.edge_index[1, i]

def test_determinism(builder):
    np.random.seed(42)
    keypoints = np.random.rand(10, 2).astype(np.float32)
    node_features = np.random.rand(10, 4).astype(np.float32)
    
    data1 = builder.build_graph(keypoints, node_features)
    data2 = builder.build_graph(keypoints, node_features)
    
    assert torch.equal(data1.edge_index, data2.edge_index)
    assert torch.equal(data1.edge_attr, data2.edge_attr)
