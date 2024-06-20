import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from hnsw_tanimoto import HNSW
from no_hnsw_tanimoto import NOHNSW
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def analyze_file(file_path):
    print(f"\nLoading molecules from {file_path}...")
    fps = load_fingerprints(file_path)
    print(f"Loaded {len(fps)} fingerprints.")

    M = 10
    hnsw = HNSW(M, Mmax=2*M, efConstruction=150, mL=1/np.log2(M), fps=fps)

    print("Starting to insert elements into HNSW...")
    start_time_insert = time.time()
    for i in tqdm(range(len(fps)), desc="Inserting into HNSW"):
        hnsw.insert(i)
    build_time = time.time() - start_time_insert
    print("Finished building HNSW structure.")

    N = 50

    print("Calculating nearest neighbors without HNSW...")
    search_nohnsw = NOHNSW(fps)
    start_time_nohnsw = time.time()
    neighbors_nohnsw = search_nohnsw.k_nearest_neighbors(0, N)
    query_time_nohnsw = time.time() - start_time_nohnsw
    print("Finished calculating neighbors without HNSW.")

    print("Calculating nearest neighbors with HNSW...")
    start_time_hnsw = time.time()
    neighbors_hnsw = hnsw.k_nn_search(0, K=N, ef=200)
    query_time_hnsw = time.time() - start_time_hnsw
    print("Finished calculating neighbors with HNSW.")

    correct_count = len(set(neighbors_hnsw).intersection(neighbors_nohnsw))
    correct_percentage = (correct_count / N) * 100

    times = {
        'build_time': build_time,
        'query_time_hnsw': query_time_hnsw,
        'query_time_nohnsw': query_time_nohnsw,
        'correct_percentage': correct_percentage,
        'algorithm_times': hnsw.time
    }
    return len(fps), times

def plot_results(results):
    dataset_sizes = [res[0] for res in results]
    build_times = [res[1]['build_time'] for res in results]
    query_times_hnsw = [res[1]['query_time_hnsw'] for res in results]
    query_times_nohnsw = [res[1]['query_time_nohnsw'] for res in results]
    correct_percentages = [res[1]['correct_percentage'] for res in results]
    algorithm_times = [res[1]['algorithm_times'] for res in results]

    plt.figure(figsize=(10, 5))
    plt.plot(dataset_sizes, build_times, marker='o', label='Build Time (minutes)')
    plt.xlabel('Dataset Size')
    plt.ylabel('Build Time (minutes)')
    plt.title('Build Time vs Dataset Size')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(dataset_sizes, query_times_hnsw, marker='o', label='HNSW Query Time (ms)')
    plt.plot(dataset_sizes, query_times_nohnsw, marker='o', label='No HNSW Query Time (ms)', linestyle='--')
    plt.xlabel('Dataset Size')
    plt.ylabel('Query Time (ms)')
    plt.title('Query Time Comparison')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(dataset_sizes, correct_percentages, marker='o', label='Correct Identification Rate (%)')
    plt.xlabel('Dataset Size')
    plt.ylabel('Correct Identification Rate (%)')
    plt.title('Correct Identification Rate vs Dataset Size')
    plt.legend()
    plt.show()

    for i, times in enumerate(algorithm_times):
        plt.figure(figsize=(10, 5))
        names = ['insert', 'search_layer', 'select_neighbors', 'k_nn_search', 'distance', 'neighborhood', 'add_connections', 'set_neighborhood']
        plt.bar(names, times)
        plt.xlabel('Algorithm Parts')
        plt.ylabel('Time (s)')
        plt.title(f'Time Spent in Each Algorithm Part for Dataset Size {dataset_sizes[i]}')
        plt.xticks(rotation=45)
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./test_illustration.py <file1> <file2> ...")
        sys.exit(1)

    results = []
    print("Starting analysis of provided files...")
    for file_path in sys.argv[1:]:
        dataset_size, times = analyze_file(file_path)
        results.append((dataset_size, times))
        print(f"Completed analysis for {file_path}")

    print("Starting to plot results...")
    plot_results(results)
    print("Finished plotting all results.")