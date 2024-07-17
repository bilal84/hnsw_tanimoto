#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <random>
#include <chrono>
#include <bitset>
#include <queue>
#include <algorithm>
#include <unordered_set>
#include <tuple>
#include <limits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <thread>
#include <mutex>
#include <future>

#include <fstream>
#include <sstream>
#include <omp.h>



bool compare_pairs(const std::pair<double, int>& a, const std::pair<double, int>& b) {
    return a.first < b.first;
}

struct pair_hash {
    std::size_t operator() (const std::pair<int, int>& pair) const {
        return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second);
    }
};



using namespace std;

constexpr size_t BITSET_SIZE = 2048;

using Fingerprint = std::bitset<BITSET_SIZE>;

double tanimoto_similarity(const Fingerprint& fp1, const Fingerprint& fp2) {
    int common_bits = (fp1 & fp2).count();
    int total_bits = fp1.count() + fp2.count() - common_bits;
    return static_cast<double>(common_bits) / total_bits;
}

struct MinHeapComp {
    bool operator()(const std::tuple<double, int, bool>& a, const std::tuple<double, int, bool>& b) const {
        return std::get<0>(a) < std::get<0>(b);
    }
};

struct MinHeapCompPair {
    bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) const {
        return std::get<0>(a) < std::get<0>(b);
    }
};

struct MaxHeapCompPair {
    bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) const {
        return std::get<0>(a) > std::get<0>(b);
    }
};






class VisitedTable {
    std::vector<bool> visited;

public:
    VisitedTable(int size) : visited(size, false) {}

    bool get(int i) {
        return visited[i];
    }

    void set(int i) {
        visited[i] = true;
    }

    void advance() {
        std::fill(visited.begin(), visited.end(), false);
    }
};

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

    Neighbor(const Neighbor& other) 
        : id(other.id), distance(other.distance), flag(other.flag) {}

    bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};


class Nhood {
public:
    std::vector<Neighbor> pool;
    std::mutex lock;

    Nhood(int L, int S, std::mt19937& rng, int ntotal) {
        pool.reserve(L);
        std::vector<int> tmp(S);
        gen_random(rng, tmp.data(), S, ntotal);
        for (int j = 0; j < S; j++) {
            int id = tmp[j];
            pool.push_back(Neighbor(id, std::numeric_limits<double>::infinity(), true));
        }
    }

    Nhood(const Nhood& other) : pool(other.pool) {
    }

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


class RNNDescent {
public:
    std::vector<Nhood> graph;
    int ntotal;
    int L = 5;
    int S = 5;
    int random_seed = 42;
    int T1 = 15;
    int T2 = 3;
    bool has_built;
    double distance_time = 0.0;
    std::vector<int> keys;
    std::vector<int> offsets;
    std::vector<int> final_graph;
    std::vector<Fingerprint> fps;

    RNNDescent(int ntotal, int L, int S, int T1, int T2, int random_seed,std::vector<int>& keys, std::vector<Fingerprint>& fps)
        : ntotal(ntotal), L(L), S(S), T1(T1), T2(T2), random_seed(random_seed), keys(keys), fps(fps), has_built(false) {}

    ~RNNDescent() {}

    double distance(int u1, int u2) {
        // auto start_distance = std::chrono::high_resolution_clock::now();

        double result = 1-tanimoto_similarity(fps[u1], fps[u2]);
        
        // auto end_distance = std::chrono::high_resolution_clock::now();
        // std::chrono::duration<double> elapsed_distance = end_distance - start_distance;
        // distance_time += elapsed_distance.count();
        return result;
    }

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

    void insert_nn(int id, int nn_id, double distance, bool flag) {
        auto& nhood = graph[id];
        {
            std::lock_guard<std::mutex> guard(nhood.lock);
            nhood.pool.emplace_back(Neighbor(nn_id, distance, flag));
        }
    }

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
        
        //std::vector<Nhood>().swap(graph);

        has_built = true;
    }
};



class HNSW {
public:
    std::vector<double> time;

    HNSW(int M, int Mmax, int efConstruction, float mL, const std::string& filename)
        : M(M), Mmax(Mmax), efConstruction(efConstruction), mL(mL), enter_point(-1), top_layer(-1) {

        layer = 0;
        
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
            fps.push_back(bs);
        }

        layers.resize(100);
        time.resize(19, 0.0);
    }

    void print_structure() const {
        for (size_t layer = 0; layer < layers.size(); ++layer) {
            if (layers[layer].empty()) continue;
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

    void insert_list(const std::vector<int>& L) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < L.size(); i++) {
            insert(i);
        }

        for (int lc = top_layer; lc >= 0; --lc) {

            std::vector<int> keys;
            int ntotal = layers[lc].size();
            keys.reserve(ntotal);
            for (const auto& pair : layers[lc]) {
                keys.push_back(pair.first);
            }

            int L = 25;
            int S = 10;
            int T1 = 5;
            int T2 = 5;

            RNNDescent rnn(ntotal, L, S, T1, T2, 42, keys, fps);

            auto start_build = std::chrono::high_resolution_clock::now();

            rnn.build(ntotal, true);

            auto end_build = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_build = end_build - start_build;
            this->time[3] += elapsed_build.count();

            this->time[2] += rnn.distance_time;

            
            auto start_save = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < rnn.graph.size(); ++i) {
                const Nhood& neighbors = rnn.graph[i];
                std::unordered_set<int> neigh = {};
                for (const Neighbor& neighbor : neighbors.pool) {
                    neigh.insert(keys[neighbor.id]);
                }
                layers[lc][keys[i]] = neigh;
            }

            auto end_save = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_save = end_save - start_save;
            this->time[4] += elapsed_save.count();
        }

        //print_structure();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[0] += elapsed.count();
    }



    void insert(int q) {
        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        int l = static_cast<int>(-log(distribution(generator)) * mL);
        int L = top_layer;
        int ep = enter_point;

        for (int lc = 0; lc <= l; ++lc) {
            layers[lc][q] = {};
        }

        if (enter_point == -1) {
            enter_point = q;
            top_layer = l;
            return;
        }

        if (l > L) {
            enter_point = q;
            top_layer = l;
        }
    }




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


    double distance(int u1, int u2) {
        auto start = std::chrono::high_resolution_clock::now();
        double result = 1-tanimoto_similarity(fps[u1], fps[u2]);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[2] += elapsed.count();
        return result;
    }




private:
    int M;
    int Mmax;
    int efConstruction;
    float mL;
    std::vector<std::unordered_map<int, std::unordered_set<int>>> layers;
    int enter_point;
    int top_layer = 0;
    std::vector<Fingerprint> fps;
    std::unordered_map<int, std::unordered_set<int>> old, new_map;
    std::unordered_map<int, std::unordered_set<int>> oldp, newp;
    int layer;

    std::unordered_map<std::pair<int, int>, std::vector<int>, pair_hash> dict;

    std::mutex counter_mutex;
    std::mutex new_map_mutex;
    std::mutex old_map_mutex;
    std::mutex v_mutex;
    std::mutex affiche;
    std::mutex insert_mutex;

    std::vector<int> neighborhood(int element, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> result;
        if (lc <= top_layer) {
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
        while (W.size()>ef) {
            W.pop();
        }

        std::vector<int> result;
        int top;
        while (W.size()>0) {
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
        .def(py::init<int, int, int, float, const std::string&>(),
             py::arg("M"), py::arg("Mmax"), py::arg("efConstruction"), py::arg("mL"), py::arg("filename"))
        .def("insert_list", &HNSW::insert_list, py::arg("indices"))
        .def("k_nn_search", &HNSW::k_nn_search, py::arg("index"), py::arg("K"), py::arg("efConstruction"))
        .def("distance", &HNSW::distance, py::arg("idx1"), py::arg("idx2"))
        .def_readwrite("time", &HNSW::time);
}