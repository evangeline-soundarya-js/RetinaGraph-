import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class RetinaGAT(torch.nn.Module):
    def __init__(self, in_channels: int = 4, hidden_channels: int = 16, num_classes: int = 2, heads: int = 4, edge_dim: int = 3, dropout: float = 0.6):
        super(RetinaGAT, self).__init__()
        self.config = {
            "in_channels": in_channels,
            "hidden_channels": hidden_channels,
            "num_classes": num_classes,
            "heads": heads,
            "edge_dim": edge_dim,
            "dropout": dropout
        }
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout, edge_dim=edge_dim)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x, alpha1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x, alpha2 = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = F.elu(x)
        
        # Global mean pooling for graph-level prediction
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x_pool = global_mean_pool(x, batch)
        
        out = self.classifier(x_pool)
        
        return out, {"alpha1": alpha1, "alpha2": alpha2}

    def load_checkpoint(self, checkpoint_path: str, device: str = "cpu"):
        """
        Loads a trained checkpoint. Currently un-utilized as the model is a prototype.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.load_state_dict(checkpoint)
