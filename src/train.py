import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.egnn import EGNN
from src.dataset import ProteinLigandDataset

def train_egnn():
    print("Initializing EGNN Training Pipeline...")
    
    # 1. Setup Data
    dataset_path = "data"
    if not os.path.exists(os.path.join(dataset_path, "labels.csv")):
        print(f"Error: Dataset not found at {dataset_path}. Run scripts/create_mock_data.py first.")
        return
        
    print(f"Loading dataset from {dataset_path}...")
    dataset = ProteinLigandDataset(root=dataset_path)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Check node feature dimension from a sample
    sample = dataset[0]
    in_node_nf = sample.x.shape[1]
    
    # 2. Setup Model
    model = EGNN(in_node_nf=in_node_nf, hidden_nf=32, out_node_nf=1, num_layers=4)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    epochs = 10
    print("Starting Training...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            
            # Forward pass (drop redundant edge_attr to fix shape mismatch, EGCL expects 1 dist dim)
            out, _ = model(batch.x, batch.pos, batch.edge_index, None, batch.batch)
            
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {epoch_loss/len(loader):.4f}")
        
    os.makedirs("models", exist_ok=True)
    model_path = "./models/egnn_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training Complete. Model saved to {model_path}")

if __name__ == "__main__":
    train_egnn()
