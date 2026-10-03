import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import Dict, Any, List

class GATExplainer:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def explain(self, data: Data, top_k: int = 10) -> Dict[str, Any]:
        """
        Extracts structural evidence (important nodes/edges) from the GAT attention weights.
        """
        with torch.no_grad():
            out, attention = self.model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=None)
            probs = F.softmax(out, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
        prediction_val = int(prediction.item())
        confidence_val = float(confidence.item())
        
        # We use alpha1 (from conv1) since it's closest to the input structural features.
        # Attention shape: tuple (edge_index, alpha)
        # alpha shape for GATConv with heads=2: [num_edges, heads]
        edge_idx, alpha = attention["alpha1"]
        
        # Aggregate multi-head attention by averaging across heads
        # This provides a unified importance score per edge
        if alpha.dim() > 1:
            edge_importance = alpha.mean(dim=1).cpu().numpy()
        else:
            edge_importance = alpha.cpu().numpy()
            
        sources = edge_idx[0].cpu().numpy()
        targets = edge_idx[1].cpu().numpy()
        
        # Compute node importance: sum of incoming attention weights
        num_nodes = data.x.shape[0]
        node_importance = np.zeros(num_nodes, dtype=np.float32)
        for i, target in enumerate(targets):
            node_importance[target] += edge_importance[i]
            
        # Top-K Nodes
        top_node_indices = np.argsort(node_importance)[::-1][:top_k]
        
        # Top-K Edges
        top_edge_indices = np.argsort(edge_importance)[::-1][:top_k]
        
        # Build Explanation Structure
        important_nodes = []
        keypoints = data.pos.cpu().numpy()
        features = data.x.cpu().numpy()
        
        for idx in top_node_indices:
            idx = int(idx)
            important_nodes.append({
                "node_index": idx,
                "importance": float(node_importance[idx]),
                "coordinates": [float(keypoints[idx, 0]), float(keypoints[idx, 1])],
                "features": {
                    "norm_x": float(features[idx, 0]),
                    "norm_y": float(features[idx, 1]),
                    "intensity": float(features[idx, 2]),
                    "vesselness": float(features[idx, 3])
                }
            })
            
        important_edges = []
        for idx in top_edge_indices:
            idx = int(idx)
            src = int(sources[idx])
            dst = int(targets[idx])
            important_edges.append({
                "edge_index": idx,
                "source": src,
                "target": dst,
                "attention_weight": float(edge_importance[idx]),
                "source_coords": [float(keypoints[src, 0]), float(keypoints[src, 1])],
                "target_coords": [float(keypoints[dst, 0]), float(keypoints[dst, 1])]
            })
            
        return {
            "prediction": prediction_val,
            "confidence": confidence_val,
            "raw_logits": out.cpu().numpy().tolist()[0],
            "method": "GAT Layer 1 Attention (Averaged across heads, summed over incoming edges)",
            "limitations": "Attention does not strictly equal clinical causality. Highlights high-message-passing structures.",
            "top_nodes": important_nodes,
            "top_edges": important_edges
        }

    def measure_faithfulness(self, data: Data, top_node_indices: List[int]) -> Dict[str, Any]:
        """
        Experiment: Mask top important nodes and measure prediction change vs random nodes.
        Masking is done by zeroing out the node features.
        """
        original_out, _ = self.model(data.x, data.edge_index, edge_attr=data.edge_attr, batch=None)
        orig_probs = F.softmax(original_out, dim=1).detach()
        orig_conf, orig_pred = torch.max(orig_probs, dim=1)
        
        # Mask Top K
        x_masked = data.x.clone()
        for idx in top_node_indices:
            x_masked[idx] = 0.0
            
        with torch.no_grad():
            top_out, _ = self.model(x_masked, data.edge_index, edge_attr=data.edge_attr, batch=None)
            top_probs = F.softmax(top_out, dim=1)
            top_conf = top_probs[0, orig_pred.item()].item()
            
        # Mask Random K
        num_nodes = data.x.shape[0]
        k = min(len(top_node_indices), num_nodes)
        random_indices = np.random.choice(num_nodes, k, replace=False)
        x_random = data.x.clone()
        for idx in random_indices:
            x_random[idx] = 0.0
            
        with torch.no_grad():
            rand_out, _ = self.model(x_random, data.edge_index, edge_attr=data.edge_attr, batch=None)
            rand_probs = F.softmax(rand_out, dim=1)
            rand_conf = rand_probs[0, orig_pred.item()].item()
            
        return {
            "experiment": "Feature Masking",
            "k_masked": k,
            "original_confidence": float(orig_conf.item()),
            "top_masked_confidence": float(top_conf),
            "random_masked_confidence": float(rand_conf),
            "confidence_drop_top": float(orig_conf.item() - top_conf),
            "confidence_drop_random": float(orig_conf.item() - rand_conf)
        }
