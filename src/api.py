import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torch_geometric.data import Batch

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.egnn import EGNN
from src.dataset import ProteinLigandDataset

app = FastAPI(title="Geometric Intelligence API")

# Add CORS to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data_real" if os.path.exists("data_real") else "data"
MODEL_PATH = "models/egnn_model.pth"

# Load the dataset (to easily access graphs) and the model globally
dataset = None
model = None

@app.on_event("startup")
def load_assets():
    global dataset, model
    print(f"Loading dataset from {DATA_PATH}...")
    dataset = ProteinLigandDataset(root=DATA_PATH)
    
    # 22 is the feature dimension (21 amino acid one-hot + 1 ligand indicator)
    in_node_nf = 22
    if len(dataset) > 0:
        in_node_nf = dataset[0].x.shape[1]
        
    print("Loading trained EGNN model...")
    model = EGNN(in_node_nf=in_node_nf, hidden_nf=64, out_node_nf=1, num_layers=4)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/complexes")
def list_complexes():
    """Returns the list of available complex IDs (e.g., 1stp, 1iep)"""
    if not os.path.exists(DATA_PATH):
        return {"complexes": []}
    
    # Exclude files like labels.csv, return only directories
    complexes = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    return {"complexes": complexes}

@app.get("/structure/{complex_id}")
def get_structure(complex_id: str):
    """Returns the raw PDB and SDF content for the frontend 3D viewer"""
    complex_dir = os.path.join(DATA_PATH, complex_id)
    if not os.path.exists(complex_dir):
        raise HTTPException(status_code=404, detail="Complex not found")
        
    protein_path = os.path.join(complex_dir, "protein.pdb")
    ligand_path = os.path.join(complex_dir, "ligand.sdf")
    
    try:
        with open(protein_path, 'r') as f:
            protein_data = f.read()
        
        ligand_data = ""
        if os.path.exists(ligand_path):
            with open(ligand_path, 'r') as f:
                ligand_data = f.read()
                
        return {
            "protein": protein_data,
            "ligand": ligand_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{complex_id}")
def predict_affinity(complex_id: str):
    """Runs a live inference pass using the trained EGNN"""
    if model is None or dataset is None:
        raise HTTPException(status_code=500, detail="Model/Dataset not loaded")
        
    # Find the index of this complex in the dataset
    target_idx = -1
    for i, data in enumerate(dataset):
        # We need a way to match data to complex_id, but our dataset might just load sequentially.
        # labels.csv has the order. Let's look up labels.csv
        import pandas as pd
        labels_path = os.path.join(DATA_PATH, "labels.csv")
        df = pd.read_csv(labels_path)
        # Find row index matching complex_id
        matches = df[df["complex_id"] == complex_id].index.tolist()
        if len(matches) > 0:
            target_idx = matches[0]
            break
            
    if target_idx == -1:
        raise HTTPException(status_code=404, detail="Complex not found in dataset labels")
        
    # Fetch the graph
    graph = dataset[target_idx]
    
    # We must batch it (size 1) to pass through the model
    batch = Batch.from_data_list([graph])
    
    with torch.no_grad():
        # Apply the exact same coordinate scaling fix as in training!
        pos = batch.pos * 0.1
        out, _ = model(batch.x, pos, batch.edge_index, None, batch.batch)
        pred_pkd = out.item()
        
    # Get actual pKd for comparison
    actual_pkd = graph.y.item()
    
    return {
        "predicted_pKd": round(pred_pkd, 3),
        "actual_pKd": round(actual_pkd, 3)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
