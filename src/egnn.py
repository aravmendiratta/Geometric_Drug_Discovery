import torch
import torch.nn as nn

class EGCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer (EGCL)
    Based on E(n) Equivariant Graph Neural Networks (Satorras et al., 2021)
    """
    def __init__(self, in_node_nf, hidden_nf, in_edge_nf=1):
        super(EGCL, self).__init__()
        self.hidden_nf = hidden_nf
        
        # Edge Multi-Layer Perceptron (computes messages)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_node_nf * 2 + in_edge_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU()
        )
        
        # Node Multi-Layer Perceptron (updates node features)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_nf + hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        
        # Coordinate Multi-Layer Perceptron (updates 3D coordinates equivariantly)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 1)
        )

    def forward(self, h, pos, edge_index, edge_attr):
        row, col = edge_index
        
        # Calculate squared distances between connected nodes
        radial = torch.sum((pos[row] - pos[col]) ** 2, dim=1).unsqueeze(1)
        
        # Combine edge attributes if provided
        if edge_attr is not None:
            radial = torch.cat([radial, edge_attr], dim=1)
            
        # 1. Message Passing (Invariant)
        edge_input = torch.cat([h[row], h[col], radial], dim=1)
        m_ij = self.edge_mlp(edge_input)
        
        # Aggregate messages for each node
        m_i = torch.zeros(h.size(0), self.hidden_nf, device=h.device)
        m_i.index_add_(0, row, m_ij)
        
        # 2. Coordinate Update (Equivariant)
        # Shift coordinates based on relative position weighted by coord_mlp output
        coord_weight = self.coord_mlp(m_ij)
        pos_diff = pos[row] - pos[col]
        pos_update = torch.zeros_like(pos)
        pos_update.index_add_(0, row, pos_diff * coord_weight)
        pos = pos + pos_update
        
        # 3. Node Feature Update (Invariant)
        node_input = torch.cat([h, m_i], dim=1)
        h = h + self.node_mlp(node_input)
        
        return h, pos

class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, num_layers=4):
        super(EGNN, self).__init__()
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.layers = nn.ModuleList([EGCL(hidden_nf, hidden_nf) for _ in range(num_layers)])
        self.readout = nn.Linear(hidden_nf, out_node_nf)
        
    def forward(self, h, pos, edge_index, edge_attr=None):
        h = self.embedding(h)
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index, edge_attr)
            
        # Global pooling (e.g., mean) to predict binding affinity
        graph_embedding = torch.mean(h, dim=0)
        out = self.readout(graph_embedding)
        return out, pos

if __name__ == "__main__":
    print("EGNN Model initialized.")
