import os
import sys
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdb_parser import parse_protein_pdb, parse_ligand_sdf
from src.graph_builder import build_protein_ligand_complex
from src.egnn import EGNN

# A small library of molecules represented as SMILES strings
# The actual target is 1stp (Streptavidin), its true ligand is Biotin.
SMILES_LIBRARY = {
    "Biotin (True Binder)": "O=C1NC2C(N1)CS[C@@H]2CCCCC(=O)O", # Known high affinity for Streptavidin
    "Imatinib (Gleevec)": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc4nccc(n4)c5cccnc5", 
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Vitamin_C": "O=C1C(=C(O)[C@@H](O)CO)O[C@@H]1O",
    "Dopamine": "c1cc(c(cc1CCN)O)O",
    "Serotonin": "c1cc2c(cc1O)c(c[nH]2)CCN",
    "Nicotine": "CN1CCCC1c2cccnc2"
}

def generate_3d_conformer(smiles: str, temp_sdf_path: str):
    """Generates a 3D conformation from a SMILES string and saves to SDF"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
        
    # Add hydrogens and generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol) # Quick forcefield relaxation
    
    writer = Chem.SDWriter(temp_sdf_path)
    writer.write(mol)
    writer.close()
    return True

def virtual_screen(target_protein_pdb: str, model_path: str):
    print(f"Loading EGNN Model from {model_path}...")
    
    # 22 is the feature dimension from our parser
    model = EGNN(in_node_nf=22, hidden_nf=64, out_node_nf=1, num_layers=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    print(f"Parsing target protein: {target_protein_pdb}")
    prot_coords, prot_features = parse_protein_pdb(target_protein_pdb)
    
    # Calculate protein center of mass for "pseudo-docking" (centering the ligand)
    prot_center = np.mean(prot_coords, axis=0)
    
    results = []
    temp_sdf = "temp_screen_ligand.sdf"
    
    print(f"\n--- Starting Virtual Screen ({len(SMILES_LIBRARY)} molecules) ---")
    
    for name, smiles in SMILES_LIBRARY.items():
        print(f"Screening {name}...")
        
        # 1. Generate 3D Conformer
        success = generate_3d_conformer(smiles, temp_sdf)
        if not success:
            continue
            
        # 2. Extract Ligand Features
        lig_coords, lig_features = parse_ligand_sdf(temp_sdf)
        
        # 3. Pseudo-docking: shift ligand center to protein center
        lig_center = np.mean(lig_coords, axis=0)
        translation_vec = prot_center - lig_center
        docked_lig_coords = lig_coords + translation_vec
        
        # 4. Build Graph
        graph = build_protein_ligand_complex(
            prot_coords, prot_features, 
            docked_lig_coords, lig_features, 
            radius=10.0 # Interaction cutoff
        )
        
        # 5. Run Inference
        batch = Batch.from_data_list([graph])
        
        with torch.no_grad():
            pos = batch.pos * 0.1 # Coordinate scaling (crucial for stability)
            out, _ = model(batch.x, pos, batch.edge_index, None, batch.batch)
            pred_pkd = out.item()
            
        results.append({"Molecule": name, "SMILES": smiles, "Predicted_pKd": pred_pkd})
        
    if os.path.exists(temp_sdf):
        os.remove(temp_sdf)
        
    # Rank results (higher pKd = stronger binding)
    df = pd.DataFrame(results)
    df = df.sort_values(by="Predicted_pKd", ascending=False).reset_index(drop=True)
    
    print("\n--- Top Hits ---")
    print(df[["Molecule", "Predicted_pKd"]].head(10))
    
    df.to_csv("screening_results.csv", index=False)
    print("\nFull results saved to screening_results.csv")

if __name__ == "__main__":
    # Screen against 1stp (Streptavidin) which was one of our real datasets
    protein_target = "data_real/1stp/protein.pdb"
    model_weights = "models/egnn_model.pth"
    
    if not os.path.exists(protein_target):
        print(f"Error: Target protein not found at {protein_target}. Please download real data first.")
    else:
        virtual_screen(protein_target, model_weights)
