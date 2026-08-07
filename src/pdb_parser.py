import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem

# Standard 20 amino acids
AMINO_ACIDS = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
               'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR']

# Common ligand atoms
ATOM_TYPES = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']

def one_hot_encode(item, vocab):
    vec = np.zeros(len(vocab) + 1) # +1 for unknown
    if item in vocab:
        vec[vocab.index(item)] = 1.0
    else:
        vec[-1] = 1.0
    return vec

def parse_protein_pdb(pdb_file_path: str):
    """
    Parses a PDB file to extract 3D coordinates and features of the protein backbone.
    Returns: coords (N, 3), features (N, 22)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    
    coords = []
    features = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    coords.append(ca_atom.get_coord())
                    
                    res_name = residue.get_resname()
                    # feature = one_hot_res (21 dims) + [is_ligand = 0]
                    feat = list(one_hot_encode(res_name, AMINO_ACIDS)) + [0.0] 
                    features.append(feat)
                    
    return np.array(coords, dtype=np.float32), np.array(features, dtype=np.float32)

def parse_ligand_sdf(sdf_file_path: str):
    """
    Parses an SDF file to extract the 3D coordinates and features of the ligand atoms.
    Returns: coords (M, 3), features (M, 22)
    """
    supplier = Chem.SDMolSupplier(sdf_file_path)
    mol = supplier[0]
    
    if mol is None:
        raise ValueError("Failed to parse ligand SDF file")
        
    conf = mol.GetConformer()
    coords = []
    features = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        
        atom_sym = atom.GetSymbol()
        
        # We need the overall feature dimension to be the same as protein (22 dims).
        # We pad atom one-hot to match the 21 dims of protein amino acids.
        feat = np.zeros(len(AMINO_ACIDS) + 1)
        atom_one_hot = one_hot_encode(atom_sym, ATOM_TYPES)
        feat[:len(atom_one_hot)] = atom_one_hot
        feat = list(feat) + [1.0] # is_ligand = 1
        
        features.append(feat)
        
    return np.array(coords, dtype=np.float32), np.array(features, dtype=np.float32)

if __name__ == "__main__":
    print("Module for parsing 3D structures from PDB and SDF files.")
