import torch
from torch_geometric.data import Data
from typing import Tuple
import numpy as np

def pure_pytorch_knn_graph(pos, k, loop=False):
    dist_matrix = torch.cdist(pos, pos)
    if not loop:
        dist_matrix.fill_diagonal_(float('inf'))
    _, col = torch.topk(-dist_matrix, k=k, dim=1)
    row = torch.arange(pos.size(0), device=pos.device).view(-1, 1).expand(-1, k)
    return torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)

def pure_pytorch_radius_graph(pos, r, loop=False):
    dist_matrix = torch.cdist(pos, pos)
    adj = dist_matrix <= r
    if not loop:
        adj.fill_diagonal_(False)
    return torch.nonzero(adj).t().contiguous()

def build_molecular_graph(coords: np.ndarray, features: np.ndarray, k: int = 5) -> Data:
    """
    Converts 3D atomic coordinates into a PyTorch Geometric Data object.
    Uses k-nearest neighbors to establish edges (bonds/proximity).
    """
    pos = torch.tensor(coords, dtype=torch.float)
    x = torch.tensor(features, dtype=torch.float)
    
    # Build k-NN graph based on 3D spatial distance
    edge_index = pure_pytorch_knn_graph(pos, k=k, loop=False)
    
    # Calculate edge attributes (e.g., initial distances)
    row, col = edge_index
    distances = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=distances, pos=pos)
    return data

def build_protein_ligand_complex(protein_coords, protein_features, ligand_coords, ligand_features, radius: float = 8.0) -> Data:
    """
    Builds a unified graph between protein pocket residues and ligand atoms using a radius graph.
    """
    # Concatenate coordinates and features
    pos = torch.cat([torch.tensor(protein_coords, dtype=torch.float), 
                     torch.tensor(ligand_coords, dtype=torch.float)], dim=0)
    x = torch.cat([torch.tensor(protein_features, dtype=torch.float), 
                   torch.tensor(ligand_features, dtype=torch.float)], dim=0)
    
    # Create edges for nodes within `radius` Angstroms of each other
    edge_index = pure_pytorch_radius_graph(pos, r=radius, loop=False)
    
    # Calculate edge attributes (Euclidean distance)
    row, col = edge_index
    distances = torch.norm(pos[row] - pos[col], p=2, dim=-1).view(-1, 1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=distances, pos=pos)
    return data

if __name__ == "__main__":
    # Example usage
    mock_coords = np.random.rand(10, 3) * 10
    mock_features = np.ones((10, 5))
    graph = build_molecular_graph(mock_coords, mock_features, k=3)
    print("Built Molecular Graph:")
    print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
