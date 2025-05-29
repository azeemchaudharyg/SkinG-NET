import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.aggr import MultiAggregation, AttentionalAggregation
from torch_geometric.nn import MessagePassing

# Custom layer to demonstrate MultiAggregation inside a GNN layer
class MultiAggrLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Combine mean, max, std aggregations, concatenate them
        aggr = MultiAggregation(['mean', 'max', 'std'], mode='cat')
        super(MultiAggrLayer, self).__init__(aggr=aggr)
        
        # Linear transform to reduce concatenated dimension back to out_channels
        self.lin = torch.nn.Linear(in_channels * 3, out_channels)  # 3 aggregations concatenated

    def forward(self, x, edge_index):
        aggr_out = self.propagate(edge_index, x=x)  # Apply aggregation
        out = self.lin(aggr_out)
        
        return out

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out
    
    

class AttentionalAggrLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        gate_nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        nn_module = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU()
        )
        aggr = AttentionalAggregation(gate_nn=gate_nn, nn=nn_module)
        super().__init__(aggr=aggr)
    
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j
