#!/usr/bin/env python3
import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from time import time
from tqdm import tqdm

build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'build'))
sys.path.append(build_path)

try:
    from hnsw_tanimoto import HNSW
except ImportError as e:
    print(f"Erreur d'importation du module hnsw_tanimoto: {e}")
    sys.exit(1)

from no_hnsw_tanimoto import NOHNSW

def save_fingerprints_to_file(file_path, output_file):
    count = 0
    with open(file_path, 'r') as file, open(output_file, 'a') as outfile:
        for line in file:
            smiles = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                bitset_str = fp.ToBitString()
                outfile.write(bitset_str + '\n')
                count += 1
    return count


def load_fingerprints(file_path):
    fingerprints = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip().split()[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                bitset_str = fp.ToBitString()
                bitset = [int(bit) for bit in bitset_str]
                fingerprints.append(bitset)
    return fingerprints



def main(active_file, inactive_file, output_file):
    N = 50

    point = 0

    M = 15
    MMax= 2*M
    efConstruction = 150
    mL = 1 / np.log2(M)


    print("WITHOUT HNSW")
    print("Loading active molecules...")
    active_fps = load_fingerprints(active_file)
    print("Loading inactive molecules...")
    inactive_fps = load_fingerprints(inactive_file)
    fps = active_fps + inactive_fps
    print("Calculating nearest neighbors without HNSW...")
    search_nohnsw = NOHNSW(fps)
    start_time_nohnsw = time()
    neighbors_nohnsw = search_nohnsw.k_nearest_neighbors(point, N)
    end_time_nohnsw = time()
    time_nohnsw = end_time_nohnsw - start_time_nohnsw
    print(f"Time taken without HNSW: {time_nohnsw} seconds")
    del search_nohnsw
    del fps


    print("\n")
    print("WITH HNSW")
    print("Saving fingerprints to file...")
    print("Saving active molecules...")
    if os.path.exists(output_file):
        os.remove(output_file)
    count_active = save_fingerprints_to_file(active_file, output_file)
    print("Saving inactive molecules...")
    count_inactive = save_fingerprints_to_file(inactive_file, output_file)
    total_fps = count_active + count_inactive
    print(f"Total fingerprints saved: {total_fps}")

    hnsw = HNSW(M, MMax, efConstruction, mL, output_file)

    print("Starting to insert elements into HNSW...")
    hnsw.insert_list([i for i in range(total_fps)])

    print("Calculating nearest neighbors with HNSW...")
    start_time_hnsw = time()
    neighbors_hnsw = hnsw.k_nn_search(point, N, efConstruction)
    end_time_hnsw = time()
    time_hnsw = end_time_hnsw - start_time_hnsw
    print(f"Time taken with HNSW: {time_hnsw} seconds")

    correct_count = sum(1 for n in neighbors_hnsw if n in neighbors_nohnsw)
    correct_percentage = (correct_count / N) * 100
    print(f"Percentage of correct neighbors found using HNSW: {correct_percentage:.2f}%")

    print(sorted(neighbors_nohnsw))
    print(sorted(list(neighbors_hnsw)))

    print("insert_list :" + str(hnsw.time[0]))
    print("k_nn_search :" + str(hnsw.time[1]))
    print("distance :" + str(hnsw.time[2]))
    print("select_neighbors :" + str(hnsw.time[3]))
    print("add_connections :" + str(hnsw.time[4]))
    print("set_neighborhood :" + str(hnsw.time[5]))
    print("neighborhood :" + str(hnsw.time[6]))
    print("search_layer :" + str(hnsw.time[7]))
    print("find_closest_elements :" + str(hnsw.time[8]))
    print("process_entry :" + str(hnsw.time[9]))
    print("insert_list_parallel :" + str(hnsw.time[10]))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: ./test.py <active_file> <inactive_file> <output_file>")
        sys.exit(1)
    
    active_file = sys.argv[1]
    inactive_file = sys.argv[2]
    output_file = sys.argv[3]
    main(active_file, inactive_file, output_file)