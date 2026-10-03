import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class RetinaGAT(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_classes: int, heads: int = 4):
        super(RetinaGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=0.6)
        self.classifier = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # We can extract attention weights for explainability
        x, alpha1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        
        x, alpha2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        
        # Global mean pooling for graph-level prediction
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        x_pool = global_mean_pool(x, batch)
        
        out = self.classifier(x_pool)
        
        return out, {"alpha1": alpha1, "alpha2": alpha2}
