import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn import MessagePassing
from models.aggregations import MultiAggrLayer, AttentionalAggrLayer

class CustomSkinCancerGNN(torch.nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, out_channels=2):
        super().__init__()

        # Layer 1: GCNConv
        self.conv1 = GCNConv(in_channels, hidden_channels)
        
        # Layer 2: GATConv with 4 heads, output size = hidden_channels
        self.conv2 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True)
        
        # Layer 3: GINConv with an MLP
        nn_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINConv(nn_mlp)
        
        # Layer 4: MultiAggregation layer
        self.multi_aggr = MultiAggrLayer(hidden_channels, hidden_channels)
        
        # Layer 5: AttentionalAggregation
        self.att_aggr = AttentionalAggrLayer(hidden_channels, hidden_channels)

        # Final classifier
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1. GCNConv
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 2. GATConv
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # 3. GINConv
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # 4. MultiAggregation layer
        x = self.multi_aggr(x, edge_index)
        x = F.relu(x)

        # 5. AttentionalAggregation layer
        x = self.att_aggr.propagate(edge_index, x=x)
        x = F.relu(x)

        # Global mean pooling to get graph-level embedding
        x = global_mean_pool(x, batch)

        # Classification
        out = self.classifier(x)
        
        return out