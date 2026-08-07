import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def create_mock_pdb(filepath, num_residues=20):
    lines = []
    residues = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
                'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
    for i in range(1, num_residues + 1):
        res = np.random.choice(residues)
        x, y, z = np.random.rand(3) * 20
        # PDB ATOM format
        line = f"ATOM  {i:5d}  CA  {res:3s} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n"
        lines.append(line)
    lines.append("END\n")
    with open(filepath, 'w') as f:
        f.writelines(lines)

def create_mock_sdf(filepath, num_atoms=15):
    # Use a basic carbon chain based on length, let RDKit generate 3D coordinates
    smiles = "C" * min(num_atoms, 20)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    writer = Chem.SDWriter(filepath)
    writer.write(mol)
    writer.close()

def main():
    os.makedirs("data", exist_ok=True)
    labels = []
    
    print("Generating mock data...")
    for i in range(1, 11):
        complex_dir = f"data/complex_{i}"
        os.makedirs(complex_dir, exist_ok=True)
        
        pdb_path = os.path.join(complex_dir, "protein.pdb")
        sdf_path = os.path.join(complex_dir, "ligand.sdf")
        
        create_mock_pdb(pdb_path, num_residues=np.random.randint(20, 50))
        create_mock_sdf(sdf_path, num_atoms=np.random.randint(10, 30))
        
        pkd = np.random.uniform(2.0, 10.0)
        labels.append({"complex_id": f"complex_{i}", "pKd": pkd})
    
    df = pd.DataFrame(labels)
    df.to_csv("data/labels.csv", index=False)
    print("Mock data generated in data/")

if __name__ == "__main__":
    main()
