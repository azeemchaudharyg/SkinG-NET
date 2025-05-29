import torch
import torch.nn as nn
from torch_geometric.nn import APPNP, global_mean_pool

class APPNPGraphClassifier(nn.Module):
    def __init__(self, K=10, alpha=0.1):
        super(APPNPGraphClassifier, self).__init__()
        
        self.lin1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.lin2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.lin3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
        self.appnp = APPNP(K=K, alpha=alpha)
        
        self.fc = nn.Linear(128, 2)

    def forward(self, x, edge_index, batch):
        x = self.dropout(self.relu(self.bn1(self.lin1(x))))
        x = self.dropout(self.relu(self.bn2(self.lin2(x))))
        x = self.dropout(self.relu(self.bn3(self.lin3(x))))
        
        x = self.appnp(x, edge_index)
        
        x_mean = global_mean_pool(x, batch)
        
        return self.fc(x_mean)