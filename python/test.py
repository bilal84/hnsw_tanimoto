import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from hnsw_tanimoto import HNSW
from no_hnsw_tanimoto import NOHNSW
import numpy as np
import time
from tqdm import tqdm

def load_fingerprints(file_path):
    fingerprints = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, 2)
                fingerprints.append(fp)
    return fingerprints

def main(active_file, inactive_file):
    print("Loading active molecules...")
    active_fps = load_fingerprints(active_file)
    print("Loading inactive molecules...")
    inactive_fps = load_fingerprints(inactive_file)
    fps = active_fps + inactive_fps

    M = 10
    hnsw = HNSW(M, Mmax=2*M, efConstruction=120, mL=1/np.log2(M), fps=fps)

    print("Starting to insert elements into HNSW...")
    for i in tqdm(range(len(fps)), desc="Inserting into HNSW"):
        hnsw.insert(i)

    N = 50

    print("Calculating nearest neighbors without HNSW...")
    search_nohnsw = NOHNSW(fps)
    start_time_nohnsw = time.time()
    neighbors_nohnsw = search_nohnsw.k_nearest_neighbors(0, N)
    end_time_nohnsw = time.time()
    time_nohnsw = end_time_nohnsw - start_time_nohnsw
    print(f"Time taken without HNSW: {time_nohnsw} seconds")

    print("Calculating nearest neighbors with HNSW...")
    start_time_hnsw = time.time()
    neighbors_hnsw = hnsw.k_nn_search(0, K=N, ef=200)
    end_time_hnsw = time.time()
    time_hnsw = end_time_hnsw - start_time_hnsw
    print(f"Time taken with HNSW: {time_hnsw} seconds")

    correct_count = 0
    for neighbor in neighbors_hnsw:
        if neighbor in neighbors_nohnsw:
            correct_count += 1

    correct_percentage = (correct_count / N) * 100
    print(f"Percentage of correct neighbors found using HNSW: {correct_percentage:.2f}%")

    print(sorted(neighbors_nohnsw))
    print(sorted(list(neighbors_hnsw)))

    print("insert :" + str(hnsw.time[0]))
    print("search_layer :" + str(hnsw.time[1]))
    print("select_neighbors :" + str(hnsw.time[2]))
    print("k_nn_search :" + str(hnsw.time[3]))
    print("distance :" + str(hnsw.time[4]))
    print("neighborhood :" + str(hnsw.time[5]))
    print("add_connections :" + str(hnsw.time[6]))
    print("set_neighborhood :" + str(hnsw.time[7]))

    #print(hnsw.layers)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: ./test.py <active_file> <inactive_file>")
        sys.exit(1)
    
    active_file = sys.argv[1]
    inactive_file = sys.argv[2]
    main(active_file, inactive_file)