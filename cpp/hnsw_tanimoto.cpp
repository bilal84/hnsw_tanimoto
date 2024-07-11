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



bool compare_pairs(const std::pair<double, int>& a, const std::pair<double, int>& b) {
    return a.first < b.first;
}

struct pair_hash {
    std::size_t operator() (const std::pair<int, int>& pair) const {
        return std::hash<int>()(pair.first) ^ std::hash<int>()(pair.second);
    }
};


constexpr size_t NEW_FPS_BITS = 128;


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

        std::vector<size_t> permutation(BITSET_SIZE);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(permutation.begin(), permutation.end(), g);

        size_t chunk_size = BITSET_SIZE / NEW_FPS_BITS;

        std::string line;
        while (std::getline(file, line)) {
            Fingerprint bs;
            for (size_t idx = 0; idx < line.length() && idx < BITSET_SIZE; ++idx) {
                if (line[idx] == '1') {
                    bs.set(idx);
                }
            }

            std::vector<char> new_fp;
            for (size_t i = 0; i < BITSET_SIZE; i += chunk_size) {
                char value = -1;
                for (size_t j = 0; j < chunk_size; ++j) {
                    if (bs[permutation[i + j]]) {
                        value = j;
                        break;
                    }
                }
                new_fp.push_back(value);
            }
            new_fps.push_back(new_fp);
        }

        layers.resize(100);
        time.resize(19, 0.0);
    }

    void process_entry(const std::vector<int>& sublist, int i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        int q = sublist[i];

        int l = static_cast<int>(-log(distribution(generator)) * mL);
        int L = top_layer;
        int ep = enter_point;

        for (int lc = L; lc > l; --lc) {
            auto W = search_layer(q, {ep}, 1, lc);
            ep = W[0];
        }

        std::vector<int> ep_vec = {ep};
        for (int lc = std::min(L, l); lc >= 0; --lc) {
            auto W = search_layer(q, ep_vec, 30, lc);
            if (lc == 0){
                auto W2 = find_closest_elements(sublist, i);

                for (int e : W2){
                    W.push_back(e);
                }
            }

            auto neighbors = select_neighbors(q, W, M, lc);

            {
                std::lock_guard<std::mutex> lock(insert_mutex);

                layers[lc][q] = {};

                dict[{lc, q}] = neighbors;
            }

            // for (int e : neighbors) {
            //     auto eConn = neighborhood(e, lc);
            //     if (eConn.size() > Mmax) {
            //         auto eNewConn = select_neighbors(e, eConn, Mmax, lc);
            //         set_neighborhood(e, eNewConn, lc);
            //         for (int neighbor : eConn) {
            //             if (std::find(eNewConn.begin(), eNewConn.end(), neighbor) == eNewConn.end()) {
            //                 layers[lc][neighbor].erase(e);
            //             }
            //         }
            //     }
            // }

            ep_vec = {};
            for (int e : W){
                ep_vec.push_back(e);
            }
        }

        {
            std::lock_guard<std::mutex> lock(insert_mutex);

            for (int lc = L; lc > l; --lc) {
                layers[lc][q] = {};
            }

            if (l > L) {
                enter_point = q;
                top_layer = l;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[9] += elapsed.count();

    }


    void insert_list_parallel(const std::vector<int>& sublist) {
        auto start = std::chrono::high_resolution_clock::now();

        constexpr size_t MAX_THREADS = 10000;
        std::vector<std::future<void>> futures;

        dict.clear();

        for (size_t i = 0; i < sublist.size(); ++i) {
            if (futures.size() >= MAX_THREADS) {
                for (auto& fut : futures) {
                    fut.wait();
                }
                futures.clear();
            }
            futures.push_back(std::async(std::launch::async, &HNSW::process_entry, this, std::ref(sublist), i));
        }
        for (auto& fut : futures) {
            fut.wait();
        }

        for (const auto& pair : dict) {
            add_connections(pair.first.second, pair.second, pair.first.first);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[10] += elapsed.count();
    }

    void insert_list(const std::vector<int>& L) {
        auto start = std::chrono::high_resolution_clock::now();

        if (!L.empty()) {
            insert(L[0]);
        }

        size_t batch_size = 100;
        for (size_t i = 1; i < L.size(); i += batch_size) {
            std::vector<int> sublist(L.begin() + i, L.begin() + std::min(L.size(), i + batch_size));

            std::cout << "test" << std::endl;
            
            insert_list_parallel(sublist);
        }

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

        for (int lc = L; lc > l; --lc) {
            auto W = search_layer(q, {ep}, 1, lc);
            ep = W[0];
        }

        std::vector<int> ep_vec = {ep};
        for (int lc = std::min(L, l); lc >= 0; --lc) {
            auto W = search_layer(q, ep_vec, efConstruction, lc);

            auto neighbors = select_neighbors(q, W, M, lc);
            add_connections(q, neighbors, lc);

            for (int e : neighbors) {
                auto eConn = neighborhood(e, lc);
                if (eConn.size() > Mmax) {
                    auto eNewConn = select_neighbors(e, eConn, Mmax, lc);
                    set_neighborhood(e, eNewConn, lc);
                    for (int neighbor : eConn) {
                        if (std::find(eNewConn.begin(), eNewConn.end(), neighbor) == eNewConn.end()) {
                            layers[lc][neighbor].erase(e);
                        }
                    }
                }
            }
            ep_vec = W;
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
            return new_fps_distance(a, q) < new_fps_distance(b, q);
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

    std::vector<std::vector<char>> new_fps;

    std::mutex counter_mutex;
    std::mutex new_map_mutex;
    std::mutex old_map_mutex;
    std::mutex v_mutex;
    std::mutex affiche;
    std::mutex insert_mutex;


    void generate_new_fingerprints(const std::vector<Fingerprint>& fps) {
        std::vector<size_t> permutation(BITSET_SIZE);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(permutation.begin(), permutation.end(), g);

        size_t chunk_size = BITSET_SIZE / NEW_FPS_BITS;

        for (const auto& bs : fps) {
            std::vector<char> new_fp;
            for (size_t i = 0; i < BITSET_SIZE; i += chunk_size) {
                char value = -1;
                for (size_t j = 0; j < chunk_size; ++j) {
                    if (bs[permutation[i + j]]) {
                        value = j;
                        break;
                    }
                }
                new_fp.push_back(value);
            }
            new_fps.push_back(new_fp);
        }
    }

    double new_fps_distance(size_t fp1_index, size_t fp2_index) {
        auto start = std::chrono::high_resolution_clock::now();

        const std::vector<char>& new_fp1 = new_fps[fp1_index];
        const std::vector<char>& new_fp2 = new_fps[fp2_index];

        int common_indices = 0;
        int common_negative = 0;

        for (size_t i = 0; i < new_fp1.size(); ++i) {
            int fp1_val = new_fp1[i], fp2_val = new_fp2[i];
            bool valid_indices = (fp1_val != -1 && fp2_val != -1);
            common_indices += (valid_indices && fp1_val == fp2_val);
            common_negative += (fp1_val == -1 && fp2_val == -1);
        }

        int adjusted_fps_bits = new_fp1.size() - common_negative;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[2] += elapsed.count();

        return adjusted_fps_bits ? 1.0 - static_cast<double>(common_indices) / adjusted_fps_bits : 1.0;
    }

    // ON VEUT DIST MIN > top pour éliminer tous ces cas
    // common indice= min des 2, common negatives = min des 2

    

    std::vector<int> select_neighbors(int q, const std::vector<int>& W, int M, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<double, int>> distances;
        for (int w : W) {
            double dist = new_fps_distance(q, w);
            distances.emplace_back(dist, w);
        }
        std::sort(distances.begin(), distances.end());

        std::vector<int> neighbors;
        for (int i = 0; i < std::min(M, static_cast<int>(distances.size())); ++i) {
            neighbors.push_back(distances[i].second);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[3] += elapsed.count();

        return neighbors;
    }


    void add_connections(int q, const std::vector<int>& neighbors, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int neighbor : neighbors) {
            layers[lc][q].insert(neighbor);
            layers[lc][neighbor].insert(q);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[4] += elapsed.count();
    }

    void set_neighborhood(int e, const std::vector<int>& new_neighbors, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        layers[lc][e] = {};

        for (int neighbor : new_neighbors) {
            layers[lc][e].insert(neighbor);
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[5] += elapsed.count();
    }


    

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
        this->time[6] += elapsed.count();

        return result;
    }


    std::vector<int> search_layer(int q, const std::vector<int>& ep, int ef, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_set<int> visited;
        std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<>> C;
        std::priority_queue<std::pair<double, int>> W;

        for (int e : ep) {
            double distance_e_q = new_fps_distance(e, q);
            C.emplace(distance_e_q, e);
            W.emplace(distance_e_q, e);
            visited.insert(e);
        }

        while (!C.empty()) {
            auto [c_dist, c_elem] = C.top();
            C.pop();
            if (W.top().first < c_dist) {
                break;
            }

            for (int e : neighborhood(c_elem, lc)) {
                if (visited.find(e) == visited.end()) {
                    double distance_e_q = new_fps_distance(e, q);
                    visited.insert(e);

                    if (W.size() < ef || distance_e_q < W.top().first) {
                        C.emplace(distance_e_q, e);
                        W.emplace(distance_e_q, e);
                        if (W.size() > ef) {
                            W.pop();
                        }
                    }
                }
            }
        }

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
        this->time[7] += elapsed.count();

        return result;
    }


    std::vector<int> find_closest_elements(const std::vector<int>& sublist, int i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::pair<double, int>> diffs;

        for (int j = 0; j < i; ++j) {
            double dist = new_fps_distance(sublist[j], sublist[i]);
            diffs.push_back({dist, j});
        }

        std::sort(diffs.begin(), diffs.end(), compare_pairs);

        std::vector<int> closest_elements;
        for (int k = 0; k < std::min(10, static_cast<int>(diffs.size())); ++k) {
            closest_elements.push_back(sublist[diffs[k].second]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[8] += elapsed.count();

        return closest_elements;
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
