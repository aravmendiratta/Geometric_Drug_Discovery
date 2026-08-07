import os
import torch
import pandas as pd
from torch_geometric.data import Dataset
from .pdb_parser import parse_protein_pdb, parse_ligand_sdf
from .graph_builder import build_protein_ligand_complex

class ProteinLigandDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """
        root: Directory containing 'labels.csv' and complex subdirectories.
        """
        self.dataset_dir = root
        self.labels_df = pd.read_csv(os.path.join(root, "labels.csv"))
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["labels.csv"]

    @property
    def processed_file_names(self):
        return []

    def len(self):
        return len(self.labels_df)

    def get(self, idx):
        row = self.labels_df.iloc[idx]
        complex_id = row['complex_id']
        pkd = row['pKd']
        
        complex_dir = os.path.join(self.dataset_dir, complex_id)
        pdb_path = os.path.join(complex_dir, "protein.pdb")
        sdf_path = os.path.join(complex_dir, "ligand.sdf")
        
        p_coords, p_feats = parse_protein_pdb(pdb_path)
        l_coords, l_feats = parse_ligand_sdf(sdf_path)
        
        data = build_protein_ligand_complex(p_coords, p_feats, l_coords, l_feats)
        data.y = torch.tensor([[pkd]], dtype=torch.float32)
        
        return data
