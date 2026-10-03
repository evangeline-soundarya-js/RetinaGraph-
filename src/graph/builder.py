import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

class GraphBuilder:
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors

    def build_graph(self, keypoints: np.ndarray, node_features: np.ndarray) -> Data:
        """
        Builds a deterministic directed KNN graph from structural keypoints.
        Generates nodes, edge_index, edge_attr, and graph structural statistics.
        """
        num_nodes = len(keypoints)
        x = torch.tensor(node_features, dtype=torch.float32)
        pos = torch.tensor(keypoints, dtype=torch.float32)
        
        sources = []
        targets = []
        edge_attrs = []
        
        # Build KNN graph based on spatial coordinates
        if num_nodes > 1:
            k = min(self.k_neighbors, num_nodes - 1)
            tree = cKDTree(keypoints)
            # Query k+1 neighbors to ensure we can find k neighbors excluding self
            distances, indices = tree.query(keypoints, k=k+1)
            
            # cKDTree might return a 1D array if k+1=1, but since k>=1, k+1>=2, it is always 2D.
            
            for i in range(num_nodes):
                valid_edges_added = 0
                for j_idx in range(k+1):
                    j = indices[i, j_idx]
                    dist = distances[i, j_idx]
                    
                    if i != j:
                        sources.append(i)
                        targets.append(j)
                        
                        # Edge features: [euclidean_distance, relative_x, relative_y]
                        rel_x = keypoints[j, 0] - keypoints[i, 0]
                        rel_y = keypoints[j, 1] - keypoints[i, 1]
                        edge_attrs.append([dist, rel_x, rel_y])
                        
                        valid_edges_added += 1
                        if valid_edges_added == k:
                            break
                        
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
        else:
            # Handle edge case with 0 or 1 node
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.float32)
            
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        
        # Calculate Graph Statistics
        num_edges = len(sources)
        if num_nodes > 0:
            avg_degree = num_edges / num_nodes
            density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0.0
            
            # Compute degrees
            in_degrees = np.zeros(num_nodes)
            out_degrees = np.zeros(num_nodes)
            for s, t in zip(sources, targets):
                out_degrees[s] += 1
                in_degrees[t] += 1
                
            total_degrees = in_degrees + out_degrees
            min_degree = int(np.min(total_degrees))
            max_degree = int(np.max(total_degrees))
            isolated_nodes = int(np.sum(total_degrees == 0))
            
            # Connected components (Weakly connected)
            if num_edges > 0:
                adj = coo_matrix((np.ones(num_edges), (sources, targets)), shape=(num_nodes, num_nodes))
                n_components, _ = connected_components(csgraph=adj, directed=False, return_labels=True)
            else:
                n_components = num_nodes
        else:
            avg_degree = 0.0
            density = 0.0
            min_degree = 0
            max_degree = 0
            isolated_nodes = 0
            n_components = 0
            
        data.stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": float(avg_degree),
            "min_degree": min_degree,
            "max_degree": max_degree,
            "density": float(density),
            "isolated_nodes": isolated_nodes,
            "connected_components": int(n_components)
        }
        
        return data
