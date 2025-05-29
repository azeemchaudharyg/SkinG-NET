import torch
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def load_features_labels(pickle_path):
    with open(pickle_path, 'rb') as f:
        features, labels = pickle.load(f)
        
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return features, labels

def split_data(features, labels, num_nodes):
    features_per_node = features.size(1) // num_nodes
    node_features = features.reshape(features.size(0), num_nodes, features_per_node)
    
    train_val_idx, test_idx = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels, random_state=42)
    y_all = torch.tensor(labels, dtype=torch.long)
    
    #train_val_labels = y_all[train_val_idx]
    #train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, stratify=train_val_labels, random_state=42)
    
    return node_features, y_all, train_val_idx, test_idx #train_idx, val_idx,

def build_data_loaders(node_features, y_all, edge_index, train_val_idx, test_idx, batch_size): #train_idx, val_idx,
    def create_loader(indices):
        return DataLoader(
            [Data(x=node_features[i], edge_index=edge_index, y=y_all[i]) for i in indices],
            batch_size=batch_size, shuffle=True
        )
    return create_loader(train_val_idx), create_loader(test_idx) #, create_loader(val_idx)