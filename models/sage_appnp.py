import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, APPNP, GINConv, GATConv, global_mean_pool, BatchNorm
from torch_geometric.nn.aggr import MultiAggregation, AttentionalAggregation
from torch_geometric.nn.aggr import MeanAggregation, MaxAggregation, StdAggregation, SumAggregation, LSTMAggregation


class CustomSAGE_APPNP(torch.nn.Module):
    def __init__(self, in_channels=512, hidden_channels=256, out_channels=2, K=10, alpha=0.1, dropout_p=0.6):
        super(CustomSAGE_APPNP, self).__init__()
        
        '''
        # Attentional Pooling
        self.attn_pool = AttentionalAggregation(
            gate_nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, 1),
                torch.nn.Sigmoid()
            )
        )
        '''
        
        
        aggr_modalities = [
            MeanAggregation(),
            #MaxAggregation(),
            StdAggregation()
        ]
        
        self.multi_aggr = MultiAggregation(
            aggr_modalities,
            mode='proj',
            mode_kwargs=dict(
                in_channels=in_channels,
                out_channels=hidden_channels,
                #num_heads=4
            )
        )

        # SAGEConv layer 1 
        self.sage1 = SAGEConv(in_channels, hidden_channels)  #, aggr=self.multi_aggr
        self.bn1 = BatchNorm(hidden_channels)
        
        '''
        # GINConv layer
        gin_nn1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.gin1 = GINConv(gin_nn1)
        self.bn_gin1 = BatchNorm(hidden_channels)
        '''

        # APPNP propagation
        self.appnp = APPNP(K=K, alpha=alpha)

        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_p)

        # Fully Connected Classifier Layer
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # SAGEConv Layer 1
        x = self.sage1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        '''
        # GINConv Layer 
        x = self.gin1(x, edge_index)
        x = self.bn_gin1(x)
        x = F.relu(x)
        x = self.dropout(x)
        '''

        # APPNP Layer
        x = self.appnp(x, edge_index)

        # Graph pooling 
        x = global_mean_pool(x, batch)
        
        # Graph pooling (Attentional Aggregation)
        #x = self.attn_pool(x, batch)

        # Fully Connected Layer
        out = self.lin(x)
        
        return out