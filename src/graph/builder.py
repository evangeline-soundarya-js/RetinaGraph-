import torch
from torch_geometric.data import Data
import numpy as np
from scipy.spatial import cKDTree

class GraphBuilder:
    def __init__(self, k_neighbors: int = 5):
        self.k_neighbors = k_neighbors

    def build_graph(self, keypoints: np.ndarray, node_features: np.ndarray) -> Data:
        """
        Builds a PyTorch Geometric Data object from keypoints and features.
        """
        # Convert to torch tensors
        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(keypoints, dtype=torch.float)
        
        # Build KNN graph based on spatial coordinates
        if len(pos) > 1:
            k = min(self.k_neighbors, len(pos) - 1)
            # Use scipy cKDTree for nearest neighbors to avoid pyg-lib requirement
            tree = cKDTree(keypoints)
            # query returns distances and indices. We query k+1 to exclude self loops
            _, indices = tree.query(keypoints, k=k+1)
            
            # Construct edge index
            sources = []
            targets = []
            for i, neighbors in enumerate(indices):
                for j in neighbors:
                    if i != j:  # skip self-loop
                        sources.append(i)
                        targets.append(j)
                        
            edge_index = torch.tensor([sources, targets], dtype=torch.long)
        else:
            # Handle edge case with 0 or 1 node
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, pos=pos)
        
        return data
