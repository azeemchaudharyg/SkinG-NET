import torch
import networkx as nx
import matplotlib.pyplot as plt

def create_fully_connected_edge_index(num_nodes):
    edges = [[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges).t().contiguous()

def visualize_graph(edge_index, num_nodes=8, seed=42):
    G = nx.Graph()
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')
    plt.title("Graph Connectivity Between Feature Nodes")
    plt.show()