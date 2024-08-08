// Standard Input/Output Stream Library
#include <iostream>
#include <fstream>
#include <sstream>

// Containers
#include <vector>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <queue>
#include <tuple>
#include <bitset>

// Algorithms
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

// Concurrency
#include <thread>
#include <mutex>
#include <future>

// OpenMP
#include <omp.h>

// Time
#include <chrono>

// Pybind
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>



using namespace std;


constexpr size_t BITSET_SIZE = 2048;

// Alias for the fingerprint using bitset
using Fingerprint = std::bitset<BITSET_SIZE>;

// Function to calculate Tanimoto similarity between two fingerprints
double tanimoto_similarity(const Fingerprint& fp1, const Fingerprint& fp2) {
    int common_bits = (fp1 & fp2).count();
    int total_bits = fp1.count() + fp2.count() - common_bits;
    return static_cast<double>(common_bits) / total_bits;
}






// Class to manage a visited table using a vector of booleans
class VisitedTable {
    std::vector<bool> visited;

public:
    // Constructor to initialize the visited table with a given size
    VisitedTable(int size) : visited(size, false) {}

    // Function to get the visited status of a specific index
    bool get(int i) {
        return visited[i];
    }

    // Function to set the visited status of a specific index to true
    void set(int i) {
        visited[i] = true;
    }

    // Function to reset all visited statuses to false
    void advance() {
        std::fill(visited.begin(), visited.end(), false);
    }
};

// Class to represent a neighbor with an ID, distance, and flag
class Neighbor {
public:
    int id;
    double distance;
    bool flag;

    // Default constructor
    Neighbor() : id(0), distance(std::numeric_limits<double>::infinity()), flag(false) {}

    // Parameterized constructor
    Neighbor(int id, double distance, bool flag)
        : id(id), distance(distance), flag(flag) {}

    // Copy constructor
    Neighbor(const Neighbor& other) 
        : id(other.id), distance(other.distance), flag(other.flag) {}

    // Operator overload for less than comparison based on distance
    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

// Class to manage a neighborhood of neighbors
class Nhood {
public:
    std::vector<Neighbor> pool;
    std::mutex lock;

    // Constructor to initialize the neighborhood with a pool of neighbors
    Nhood(int L, int S, std::mt19937& rng, int ntotal) {
        pool.reserve(L);
        std::vector<int> tmp(S);
        gen_random(rng, tmp.data(), S, ntotal);
        for (int j = 0; j < S; j++) {
            int id = tmp[j];
            pool.push_back(Neighbor(id, std::numeric_limits<double>::infinity(), true));  // Distance and flag will be updated later
        }
    }

    // Copy constructor
    Nhood(const Nhood& other) : pool(other.pool) {
        // Note that we do not need to copy the mutex
    }

    // Static function to generate random indices for the neighbors
    static void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {
        for (int i = 0; i < size; ++i) {
            addr[i] = rng() % (N - size);
        }
        std::sort(addr, addr + size);
        for (int i = 1; i < size; ++i) {
            if (addr[i] <= addr[i - 1]) {
                addr[i] = addr[i - 1] + 1;
            }
        }
        int off = rng() % N;
        for (int i = 0; i < size; ++i) {
            addr[i] = (addr[i] + off) % N;
        }
    }
};


// Class for RNNDescent algorithm
class RNNDescent {
public:
    std::vector<Nhood> graph;  // Vector to store the graph's neighborhoods
    int ntotal;  // Total number of nodes
    int L = 25;  // Maximum number of neighbors
    int S = 10;  // Initial number of neighbors
    int random_seed = 42;  // Seed for random number generator
    int T1 = 5;  // Number of iterations for the outer loop
    int T2 = 5;  // Number of iterations for the inner loop
    bool has_built;  // Flag to indicate if the graph has been built
    double distance_time = 0.0;  // Variable to store the time taken for distance calculations
    std::vector<int> keys;  // Vector to store keys
    std::vector<int> offsets;  // Vector to store offsets
    std::vector<int> final_graph;  // Vector to store the final graph
    std::vector<Fingerprint> fps;  // Vector to store fingerprints

    // Constructor
    RNNDescent(int ntotal, int L, int S, int T1, int T2, int random_seed, std::vector<int>& keys, std::vector<Fingerprint>& fps)
        : ntotal(ntotal), L(L), S(S), T1(T1), T2(T2), random_seed(random_seed), keys(keys), fps(fps), has_built(false) {}

    // Destructor to free memory used by dynamic structures
    ~RNNDescent() {
        for (auto& nhood : graph) {
            nhood.pool.clear();
        }
        graph.clear();
        keys.clear();
        offsets.clear();
        final_graph.clear();
        fps.clear();
    }

    // Function to calculate distance between two nodes based on Tanimoto similarity
    double distance(int u1, int u2) {
        double result = 1 - tanimoto_similarity(fps[u1], fps[u2]);
        return result;
    }

    // Function to initialize the graph
    void init_graph() {
        graph.reserve(ntotal);
        {
            std::mt19937 rng(random_seed * 6007);
            for (int i = 0; i < ntotal; i++) {
                graph.push_back(Nhood(L, S, rng, (int)ntotal));
            }
        }

        #pragma omp parallel
        {
            std::mt19937 rng(random_seed * 7741 + std::hash<std::thread::id>{}(std::this_thread::get_id()));
            #pragma omp for
            for (int i = 0; i < ntotal; i++) {
                std::vector<int> tmp(S);
                Nhood::gen_random(rng, tmp.data(), S, ntotal);
                for (int j = 0; j < S; j++) {
                    int id = tmp[j];
                    if (id == i) continue;
                    double dist = distance(keys[i], keys[id]);
                    graph[i].pool.push_back(Neighbor(id, dist, true));
                }
                std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
                graph[i].pool.reserve(L);
            }
        }
    }

    // Function to insert a nearest neighbor
    void insert_nn(int id, int nn_id, double distance, bool flag) {
        auto& nhood = graph[id];
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            nhood.pool.emplace_back(Neighbor(nn_id, distance, flag));
        }
    }

    // Function to update neighbors
    void update_neighbors() {
        #pragma omp parallel for schedule(dynamic, 256)
        for (int u = 0; u < ntotal; ++u) {
            auto& nhood = graph[u];
            auto& pool = nhood.pool;
            std::vector<Neighbor> new_pool;
            std::vector<Neighbor> old_pool;
            {
                std::lock_guard<std::mutex> guard(nhood.lock);
                old_pool = pool;
                pool.clear();
            }
            std::sort(old_pool.begin(), old_pool.end());
            old_pool.erase(std::unique(old_pool.begin(), old_pool.end(),
                                       [](Neighbor& a, Neighbor& b) {
                                           return a.id == b.id;
                                       }),
                           old_pool.end());

            for (auto&& nn : old_pool) {
                bool ok = true;
                for (auto&& other_nn : new_pool) {
                    if (!nn.flag && !other_nn.flag) {
                        continue;
                    }
                    if (nn.id == other_nn.id) {
                        ok = false;
                        break;
                    }
                    double dist = distance(keys[nn.id], keys[other_nn.id]);
                    if (dist < nn.distance) {
                        ok = false;
                        insert_nn(other_nn.id, nn.id, dist, true);
                        break;
                    }
                }
                if (ok) {
                    new_pool.emplace_back(nn);
                }
            }

            for (auto&& nn : new_pool) {
                nn.flag = false;
            }
            {
                std::lock_guard<std::mutex> guard(nhood.lock);
                pool.insert(pool.end(), new_pool.begin(), new_pool.end());
            }
        }
    }

    // Function to add reverse edges to the graph
    void add_reverse_edges() {
        std::vector<std::vector<Neighbor>> reverse_pools(ntotal);

        #pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            for (auto&& nn : graph[u].pool) {
                std::lock_guard<std::mutex> guard(graph[nn.id].lock);
                reverse_pools[nn.id].emplace_back(u, nn.distance, nn.flag);
            }
        }

        #pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            auto& pool = graph[u].pool;
            for (auto&& nn : pool) {
                nn.flag = true;
            }
            auto& rpool = reverse_pools[u];
            rpool.insert(rpool.end(), pool.begin(), pool.end());
            pool.clear();
            std::sort(rpool.begin(), rpool.end());
            rpool.erase(std::unique(rpool.begin(), rpool.end(),
                                    [](Neighbor& a, Neighbor& b) {
                                        return a.id == b.id;
                                    }),
                        rpool.end());
            if (rpool.size() > L) {
                rpool.resize(L);
            }
        }

        #pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            for (auto&& nn : reverse_pools[u]) {
                std::lock_guard<std::mutex> guard(graph[nn.id].lock);
                graph[nn.id].pool.emplace_back(u, nn.distance, nn.flag);
            }
        }

        #pragma omp parallel for
        for (int u = 0; u < ntotal; ++u) {
            auto& pool = graph[u].pool;
            std::sort(pool.begin(), pool.end());
            if (pool.size() > L) {
                pool.resize(L);
            }
        }
    }

    // Function to build the graph
    void build(const int n, bool verbose) {
        if (verbose) {
            printf("Parameters: S=%d, L=%d\n", S, L);
        }

        ntotal = n;
        init_graph();

        for (int t1 = 0; t1 < T1; ++t1) {
            if (verbose) {
                std::cout << "Iter " << t1 << " : " << std::flush;
            }
            for (int t2 = 0; t2 < T2; ++t2) {
                update_neighbors();
                if (verbose) {
                    std::cout << "#" << std::flush;
                }
            }

            if (t1 != (T1-1)) {
                add_reverse_edges();
            }

            if (verbose) {
                printf("\n");
            }
        }

        #pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            auto& pool = graph[u].pool;
            std::sort(pool.begin(), pool.end());
            pool.erase(std::unique(pool.begin(), pool.end(),
                                   [](Neighbor& a, Neighbor& b) {
                                       return a.id == b.id;
                                   }),
                       pool.end());
        }

        offsets.resize(ntotal + 1);
        offsets[0] = 0;
        for (int u = 0; u < ntotal; ++u) {
            offsets[u + 1] = offsets[u] + graph[u].pool.size();
        }

        final_graph.resize(offsets.back(), -1);
        #pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            auto& pool = graph[u].pool;
            int offset = offsets[u];
            for (int i = 0; i < pool.size(); ++i) {
                final_graph[offset + i] = pool[i].id;
            }
        }

        has_built = true;
    }
};



// Class for HNSW (Hierarchical Navigable Small World) structure
class HNSW {
public:
    int total_fps = 0;  // Total number of fingerprints

    // Paramaters for RNNDescent algorithm
    int L = 25;  // Maximum number of neighbors
    int S = 10;  // Initial number of neighbors
    int T1 = 5;  // Number of iterations for the outer loop
    int T2 = 5;  // Number of iterations for the inner loop

    std::vector<double> time;  // Vector to store timing information

    // Constructor
    HNSW(float mL, int L, int S, int T1, int T2, const std::string& filename)
        : mL(mL), L(L), S(S), T1(T1), T2(T2), enter_point(-1), top_layer(-1) {
        
        std::cout << "Initializing HNSW structure from file: " << filename << std::endl;

        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            Fingerprint bs;
            for (size_t idx = 0; idx < line.length() && idx < BITSET_SIZE; ++idx) {
                if (line[idx] == '1') {
                    bs.set(idx);
                }
            }
            total_fps++;
            fps.push_back(bs);
            std::cout << total_fps << std::endl;
        }

        std::cout << total_fps << std::endl;

        layers.resize(100);  // Adjust the size of the layers as necessary
        time.resize(19, 0.0);
    }

    // Function to print the structure of the HNSW
    void print_structure() const {
        for (size_t layer = 0; layer < layers.size(); ++layer) {
            if (layers[layer].empty()) continue; // Skip empty layers
            std::cout << "Layer " << layer << ":\n";
            for (const auto& element : layers[layer]) {
                std::cout << "  Node " << element.first << ": Neighbors: ";
                for (const auto& neighbor : element.second) {
                    std::cout << neighbor << ", ";
                }
                std::cout << "\n";
            }
        }
    }

    // Function to insert a list of elements
    void insert_list(const std::vector<int>& V, int num_threads) {
        auto start = std::chrono::high_resolution_clock::now();

        omp_set_num_threads(num_threads);

        // Insert elements
        for (int i = 0; i < V.size(); ++i) {
            insert(i);
        }

        // Insertion in each layer
        for (int lc = top_layer; lc >= 0; --lc) {
            std::vector<int> keys;
            int ntotal = layers[lc].size();
            keys.reserve(ntotal);
            for (const auto& pair : layers[lc]) {
                keys.push_back(pair.first);
            }

            std::cout << "Number of elements to insert in the layer " << lc << ": " << ntotal << std::endl;

            if (ntotal < 30) {
                // Naive connection method
                for (int i = 0; i < ntotal; ++i) {
                    std::unordered_set<int> neigh;
                    for (int j = 0; j < ntotal; ++j) {
                        if (i != j) {
                            neigh.insert(keys[j]);
                        }
                    }
                    layers[lc][keys[i]] = neigh;
                }
            } else {
                // Graph construction for each layer
                RNNDescent* rnn = new RNNDescent(ntotal, L, S, T1, T2, 42, keys, fps);

                auto start_build = std::chrono::high_resolution_clock::now();

                rnn->build(ntotal, true);

                auto end_build = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_build = end_build - start_build;
                this->time[3] += elapsed_build.count();

                this->time[2] += rnn->distance_time;

                auto start_save = std::chrono::high_resolution_clock::now();

                // Store the results in layers
                for (size_t i = 0; i < rnn->graph.size(); ++i) {
                    const Nhood& neighbors = rnn->graph[i];
                    std::unordered_set<int> neigh = {};
                    for (const Neighbor& neighbor : neighbors.pool) {
                        neigh.insert(keys[neighbor.id]);
                    }
                    layers[lc][keys[i]] = neigh;
                }

                delete rnn;

                auto end_save = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_save = end_save - start_save;
                this->time[4] += elapsed_save.count();
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[0] += elapsed.count();  // Add the total operation time to a time accumulator
    }

    // Function to insert an element
    void insert(int q) {
        static std::default_random_engine generator(std::random_device{}());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        int l = static_cast<int>(-log(distribution(generator)) * mL);
        int L = top_layer;
        int ep = enter_point;

        // Add q to all levels up to level l
        for (int lc = 0; lc <= l; ++lc) {
            layers[lc][q] = {};
        }

        // Case for the first inserted element
        if (enter_point == -1) {
            enter_point = q;
            top_layer = l;
            return;
        }

        // Update the enter point, which is the point with the highest layer (top_layer)
        if (l > L) {
            enter_point = q;
            top_layer = l;
        }
    }

    // Function to perform k-nearest neighbors search
    std::set<int> k_nn_search(int q, int K, int ef) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> W;
        int ep = enter_point;
        int L = top_layer;
        
        for (int lc = L; lc > 0; --lc) {
            W = search_layer(q, {ep}, 1, lc);
            ep = W[0];
        }

        W = search_layer(q, {ep}, ef, 0);

        std::sort(W.begin(), W.end(), [this, q](int a, int b) {
            return distance(a, q) < distance(b, q);
        });

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[1] += elapsed.count();

        std::set<int> result(W.begin() + 1, W.begin() + std::min(K + 1, static_cast<int>(W.size())));

        return result;
    }

    // Function to calculate distance between two nodes based on Tanimoto similarity
    double distance(int u1, int u2) {
        auto start = std::chrono::high_resolution_clock::now();
        double result = 1 - tanimoto_similarity(fps[u1], fps[u2]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[2] += elapsed.count();
        return result;
    }

private:
    float mL;  // Level multiplier
    std::vector<std::unordered_map<int, std::unordered_set<int>>> layers;  // Vector of maps representing the layers
    int enter_point;  // Entry point for the search
    int top_layer = 0;  // Highest layer level
    std::vector<Fingerprint> fps;  // Vector to store fingerprints

    // Function to get the neighborhood of an element in a specific layer
    std::vector<int> neighborhood(int element, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> result;
        if (lc <= top_layer) {  // Ensure comparison up to `top_layer`
            auto it = layers[lc].find(element);
            if (it != layers[lc].end()) {
                for (const auto& neighbor : it->second) {
                    result.push_back(neighbor);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[5] += elapsed.count();

        return result;
    }

    // Function to search within a layer
    std::vector<int> search_layer(int q, const std::vector<int>& ep, int ef, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_set<int> visited;
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> C;
        std::priority_queue<std::pair<double, int>> W;

        // Initial insertion of elements from ep into heaps
        for (int e : ep) {
            double distance_e_q = distance(e, q);
            C.emplace(distance_e_q, e);
            W.emplace(distance_e_q, e);
            visited.insert(e);
        }

        while (!C.empty()) {
            auto [c_dist, c_elem] = C.top();
            C.pop();
            if (W.top().first < c_dist) {
                break;  // All elements in W are evaluated
            }

            for (int e : neighborhood(c_elem, lc)) {
                if (visited.find(e) == visited.end()) {
                    double distance_e_q = distance(e, q);
                    visited.insert(e);

                    if (W.size() < ef || distance_e_q < W.top().first) {
                        C.emplace(distance_e_q, e);
                        W.emplace(distance_e_q, e);
                        if (W.size() > ef) {
                            W.pop();  // Remove the furthest element
                        }
                    }
                }
            }
        }

        // Collect the ef elements with the smallest distances
        while (W.size() > ef) {
            W.pop();
        }

        std::vector<int> result;
        int top;
        while (W.size() > 0) {
            top = W.top().second;
            W.pop();
            result.push_back(top);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[6] += elapsed.count();

        return result;
    }
};



namespace py = pybind11;

PYBIND11_MODULE(hnsw_tanimoto, m) {
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<float, int, int, int, int, const std::string&>(),
             py::arg("mL"),
             py::arg("L"), py::arg("S"), py::arg("T1"), py::arg("T2"), py::arg("filename"))
        .def("insert_list", &HNSW::insert_list, py::arg("indices"), py::arg("num_threads"))
        .def("k_nn_search", &HNSW::k_nn_search, py::arg("index"), py::arg("K"), py::arg("efConstruction"))
        .def("distance", &HNSW::distance, py::arg("idx1"), py::arg("idx2"))
        .def_readwrite("time", &HNSW::time);
}
