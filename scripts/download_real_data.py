import os
import urllib.request
import pandas as pd
from rdkit import Chem

# Known PDB IDs and their approximate binding affinities (pKd)
# 1stp: Streptavidin + Biotin (very strong)
# 1iep: Abl kinase + Imatinib (strong)
# 1a28: Progesterone receptor + Progesterone
# 3ptb: Trypsin + Benzamidine (moderate)
COMPLEXES = {
    "1stp": {"pkd": 13.5, "ligand_name": "BTN"},
    "1iep": {"pkd": 7.6,  "ligand_name": "STI"},
    "1a28": {"pkd": 7.3,  "ligand_name": "STR"},
    "3ptb": {"pkd": 4.5,  "ligand_name": "BEN"}
}

def download_pdb(pdb_id, output_path):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    urllib.request.urlretrieve(url, output_path)

def extract_protein_and_ligand(pdb_path, out_protein_pdb, out_ligand_sdf, ligand_resname):
    # Read the downloaded PDB
    with open(pdb_path, 'r') as f:
        lines = f.readlines()
        
    # 1. Save protein (ATOM records)
    protein_lines = [line for line in lines if line.startswith("ATOM")]
    with open(out_protein_pdb, 'w') as f:
        f.writelines(protein_lines)
        f.write("END\n")
        
    # 2. Extract ligand (HETATM records matching the ligand name)
    ligand_lines = [line for line in lines if line.startswith("HETATM") and ligand_resname in line]
    
    # We write a temporary PDB for the ligand to let RDKit convert it to SDF
    temp_ligand_pdb = pdb_path.replace(".pdb", "_lig_temp.pdb")
    with open(temp_ligand_pdb, 'w') as f:
        f.writelines(ligand_lines)
        f.write("END\n")
        
    # Use RDKit to read the ligand PDB and write it as SDF
    mol = Chem.MolFromPDBFile(temp_ligand_pdb)
    if mol is not None:
        writer = Chem.SDWriter(out_ligand_sdf)
        writer.write(mol)
        writer.close()
    else:
        print(f"Warning: RDKit could not parse ligand for {pdb_path}")
        # Create an empty SDF to prevent crashes
        with open(out_ligand_sdf, 'w') as f:
            f.write("")
            
    # Cleanup temp file
    if os.path.exists(temp_ligand_pdb):
        os.remove(temp_ligand_pdb)

def main():
    data_dir = "data_real"
    os.makedirs(data_dir, exist_ok=True)
    labels = []
    
    print("Downloading and processing real PDB complexes...")
    
    for pdb_id, info in COMPLEXES.items():
        print(f"Processing {pdb_id} (Ligand: {info['ligand_name']})...")
        complex_dir = os.path.join(data_dir, pdb_id)
        os.makedirs(complex_dir, exist_ok=True)
        
        raw_pdb_path = os.path.join(complex_dir, f"{pdb_id}_raw.pdb")
        protein_path = os.path.join(complex_dir, "protein.pdb")
        ligand_path = os.path.join(complex_dir, "ligand.sdf")
        
        # Download
        download_pdb(pdb_id, raw_pdb_path)
        
        # Process and split
        extract_protein_and_ligand(raw_pdb_path, protein_path, ligand_path, info['ligand_name'])
        
        # Save label
        labels.append({"complex_id": pdb_id, "pKd": info['pkd']})
        
        # Cleanup raw download
        if os.path.exists(raw_pdb_path):
            os.remove(raw_pdb_path)
            
    # Write labels
    df = pd.DataFrame(labels)
    df.to_csv(os.path.join(data_dir, "labels.csv"), index=False)
    print(f"Real data successfully processed and saved to '{data_dir}/'")

if __name__ == "__main__":
    main()
