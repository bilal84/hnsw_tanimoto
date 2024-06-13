import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from hnsw_tanimoto import HNSW
import numpy as np

def load_molecules(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    molecules = []
    for line in lines:
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molecules.append(mol)
    return molecules

def create_fingerprints(molecules):
    fingerprints = [AllChem.GetMorganFingerprint(mol, 2) for mol in molecules]
    return fingerprints

def main(active_file, inactive_file):
    active_molecules = load_molecules(active_file)
    inactive_molecules = load_molecules(inactive_file)
    molecules = active_molecules+inactive_molecules

    active_fps = create_fingerprints(active_molecules)
    inactive_fps = create_fingerprints(inactive_molecules)
    fps = active_fps + inactive_fps

    M=5
    hnsw = HNSW(M, Mmax=2*M, efConstruction=15, mL=1/np.log2(M), molecules=molecules, fps=fps)

    for i in range(len(active_molecules + inactive_molecules)):
        print(str(i))
        hnsw.insert(i)

    N = len(active_molecules)
    neighbors_mol = hnsw.k_nn_search(0, K=N-1, ef=N-1) # we look at the neighbors of molecule 0, which is (in this case) an active molecule


    active_count = 0
    for neighbor_mol in neighbors_mol:
        if any(AllChem.DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(molecules[neighbor_mol], 2), active_fp) == 1.0 for active_fp in active_fps):
            active_count += 1

    active_percentage = (active_count / (N-1)) * 100
    print(f"Percentage of active compounds among the nearest neighbors: {active_percentage:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./test.py <active_file> <inactive_file>")
        sys.exit(1)
    
    active_file = sys.argv[1]
    inactive_file = sys.argv[2]
    main(active_file, inactive_file)
