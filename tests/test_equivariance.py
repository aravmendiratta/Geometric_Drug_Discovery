import torch
import math
from src.egnn import EGNN

def test_se3_equivariance():
    """
    Tests if the EGNN model is SE(3) equivariant/invariant.
    Specifically, the predicted binding affinity (invariant) should remain identical
    if the input 3D coordinates are rotated.
    """
    print("Running SE(3) Equivariance Test...")
    
    # 1. Setup Model
    model = EGNN(in_node_nf=5, hidden_nf=32, out_node_nf=1, num_layers=3)
    model.eval()
    
    # 2. Create a mock input graph
    num_nodes = 10
    h = torch.randn(num_nodes, 5)
    pos = torch.randn(num_nodes, 3)
    row, col = torch.combinations(torch.arange(num_nodes), r=2).T
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    
    # 3. Define a 3D Rotation Matrix (e.g., 90 degrees around Z axis)
    theta = math.radians(90)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rotation_matrix = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t, cos_t, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Apply rotation to positions
    rotated_pos = torch.matmul(pos, rotation_matrix.T)
    
    # 4. Forward Passes
    with torch.no_grad():
        out_original, pos_updated_original = model(h, pos, edge_index)
        out_rotated, pos_updated_rotated = model(h, rotated_pos, edge_index)
        
    # 5. Assertions
    # Invariant output (binding affinity) should be identical
    assert torch.allclose(out_original, out_rotated, atol=1e-5), \
        "Model invariant output is NOT SE(3) invariant!"
        
    # Equivariant output (updated positions) should be rotated by the same matrix
    rotated_pos_updated_original = torch.matmul(pos_updated_original, rotation_matrix.T)
    assert torch.allclose(rotated_pos_updated_original, pos_updated_rotated, atol=1e-5), \
        "Model coordinate output is NOT SE(3) equivariant!"
        
    print("SUCCESS: EGNN Model mathematically proven to be SE(3) Equivariant/Invariant!")

if __name__ == "__main__":
    test_se3_equivariance()
