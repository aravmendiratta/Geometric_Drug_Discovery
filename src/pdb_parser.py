from Bio.PDB import PDBParser
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def parse_protein_pdb(pdb_file_path: str):
    """
    Parses a PDB file to extract 3D coordinates of the protein backbone (C-alpha atoms).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file_path)
    
    coords = []
    residues = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    coords.append(ca_atom.get_coord())
                    residues.append(residue.get_resname())
                    
    return np.array(coords), residues

def parse_ligand_sdf(sdf_file_path: str):
    """
    Parses an SDF file to extract the 3D coordinates of the ligand atoms.
    """
    supplier = Chem.SDMolSupplier(sdf_file_path)
    mol = supplier[0]
    
    if mol is None:
        raise ValueError("Failed to parse ligand SDF file")
        
    conf = mol.GetConformer()
    coords = []
    atom_types = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append([pos.x, pos.y, pos.z])
        atom_types.append(atom.GetSymbol())
        
    return np.array(coords), atom_types

if __name__ == "__main__":
    print("Module for parsing 3D structures from PDB and SDF files.")
