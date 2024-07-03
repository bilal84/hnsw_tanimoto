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



using namespace std;

constexpr size_t BITSET_SIZE = 2048;

using Fingerprint = std::bitset<BITSET_SIZE>;

double tanimoto_similarity(const Fingerprint& fp1, const Fingerprint& fp2) {
    int common_bits = (fp1 & fp2).count();
    int total_bits = fp1.count() + fp2.count() - common_bits;
    return static_cast<double>(common_bits) / total_bits;
}

class HNSW {
public:
    std::vector<double> time;

    HNSW(int K, int efConstruction, float mL, const std::string& filename)
        : K(K), efConstruction(efConstruction), mL(mL), enter_point(-1), top_layer(-1) {

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
                    double priority;
                    int node;
                    bool flag;
                    std::tie(priority, node, flag) = neighbor;
                    std::cout << "(" << priority << ", " << node << ", " << flag << ") ";
                }
                std::cout << "\n";
            }
        }
    }




    void insert_list(const std::vector<int>& L) {
        auto start = std::chrono::high_resolution_clock::now();

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (int q : L) {
            int l = static_cast<int>(-log(distribution(generator)) * mL);
            if (enter_point == -1) {
                enter_point = q;
                top_layer = l;
            }
            for (int lc = 0; lc <= l; ++lc) {
                std::vector<std::tuple<double, int, bool>> empty_heap;
                layers[lc][q] = empty_heap;
            }
            if (l > top_layer) {
                enter_point = q;
                top_layer = l;
            }
        }

        for (int l = 0; l < layers.size(); ++l) {
            layer = l;
            auto B = layers[l];
            layers[l] = nn_descent_full(B, K, l);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[0] += elapsed.count();
    }


    std::set<int> k_nn_search(int q, int K, int ef) {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> W;
        int ep = enter_point;
        int L = top_layer;
        
        for (int lc = L; lc > 0; --lc) {
            W = searchLayer(q, {ep}, 1, lc);
            ep = W[0];
        }

        W = searchLayer(q, {ep}, ef, 0);

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
    int efConstruction;
    float mL;
    std::vector<std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>> layers;
    int enter_point;
    int top_layer;
    std::vector<Fingerprint> fps;
    std::unordered_map<int, std::vector<std::pair<double, int>>> old, new_map;
    std::unordered_map<int, std::vector<std::pair<double, int>>> oldp, newp;
    int layer;
    int K;

    std::mutex counter_mutex;
    std::mutex new_map_mutex;
    std::mutex old_map_mutex;
    std::mutex v_mutex;
    std::mutex index_mutex;


    void process_entry(std::pair<const int, std::vector<std::tuple<double, int, bool>>>& v,
                   int rho) {
        auto start = std::chrono::high_resolution_clock::now();
        {
            for (const auto& item : v.second) {
                double d; int u; bool flag;
                std::tie(d, u, flag) = item;
                if (!flag) {
                    std::lock_guard<std::mutex> lock(old_map_mutex);
                    old[v.first].emplace_back(d, u);
                }
            }
        }

        std::vector<std::pair<double, int>> sampled;
        {
            std::lock_guard<std::mutex> lock(new_map_mutex);
            sampled = sample_full2(new_map[v.first], static_cast<int>(rho * K));
            new_map[v.first] = sampled;
        }

        std::unordered_set<int> sampledSet;
        for (const auto& pair : sampled) {
            sampledSet.insert(pair.second);
        }

        for (auto& item : v.second) {
            int id = std::get<1>(item);
            if (sampledSet.find(id) != sampledSet.end()) {
                std::get<2>(item) = false;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[13] += elapsed.count();
    }

    void parallel_for(std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& B, 
                      int rho) {
        auto start = std::chrono::high_resolution_clock::now();

        constexpr size_t MAX_THREADS = 10000;
        std::vector<std::future<void>> futures;

        for (auto& v : B) {
            if (futures.size() >= MAX_THREADS) {
                for (auto& fut : futures) {
                    fut.wait();
                }
                futures.clear();
            }
            futures.push_back(std::async(std::launch::async, &HNSW::process_entry, this, std::ref(v), rho));
        }

        for (auto& fut : futures) {
            fut.wait();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[14] += elapsed.count();
    }



    void process_entry2(
    const std::pair<const int, std::vector<std::tuple<double, int, bool>>>& v, 
    std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& B,
    int rho, int layer, int& counter) 
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> oldp_keys;
        if (oldp.find(v.first) != oldp.end()) {
            for (const auto& pair : oldp[v.first]) {
                oldp_keys.push_back(pair.second);
            }
        }
        {
            std::vector<std::pair<double, int>> old_samples = sample(oldp_keys, static_cast<int>(rho * K), -1);
            std::lock_guard<std::mutex> old_lock(old_map_mutex);
            old[v.first] = merge_heaps_full(old[v.first], old_samples);
            std::make_heap(old[v.first].begin(), old[v.first].end());
        }

        std::vector<int> newp_keys;
        if (newp.find(v.first) != newp.end()) {
            for (const auto& pair : newp[v.first]) {
                newp_keys.push_back(pair.second);
            }
        }
        {
            std::vector<std::pair<double, int>> new_samples = sample(newp_keys, static_cast<int>(rho * K), -1);
            std::lock_guard<std::mutex> new_lock(new_map_mutex);
            new_map[v.first] = merge_heaps_full(new_map[v.first], new_samples);
            std::make_heap(new_map[v.first].begin(), new_map[v.first].end());
        }

        for (const auto& [d1, u1] : new_map[v.first]) {
            for (const auto& [d2, u2] : new_map[v.first]) {
                if (u1 < u2) {
                    double l = distance(u1, u2);
                    std::lock_guard<std::mutex> counter_lock(counter_mutex);
                    counter += update_nn_full(B[u1], std::make_tuple(l, u2, true), layer, u1, K);
                    counter += update_nn_full(B[u2], std::make_tuple(l, u1, true), layer, u2, K);
                }
            }
            for (const auto& [d2, u2] : old[v.first]) {
                if (u1 < u2) {
                    double l = distance(u1, u2);
                    std::lock_guard<std::mutex> counter_lock(counter_mutex);
                    counter += update_nn_full(B[u1], std::make_tuple(l, u2, true), layer, u1, K);
                    counter += update_nn_full(B[u2], std::make_tuple(l, u1, true), layer, u2, K);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[15] += elapsed.count();
    }

    
    void parallel_for2(
    std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& B,
    int rho, int layer, int& counter,
    int batchSize = 10000)
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::future<void>> futures;
        std::vector<std::pair<const int, std::vector<std::tuple<double, int, bool>>>> items(B.begin(), B.end());

        for (size_t i = 0; i < items.size(); i += batchSize) {
            auto end = std::min(items.size(), i + batchSize);

            for (size_t j = i; j < end; ++j) {
                futures.push_back(std::async(std::launch::async, &HNSW::process_entry2, this, std::cref(items[j]),
                                            std::ref(B), rho, layer, std::ref(counter)));
            }

            for (auto& fut : futures) {
                fut.get();
            }
            futures.clear();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[16] += elapsed.count();
    }

    void process_entry3(std::vector<int>& keys,
    std::pair<const int, std::vector<std::tuple<double, int, bool>>>& v, int v_first){
        auto start = std::chrono::high_resolution_clock::now();

        auto samples = sample_full(keys, v_first);
        std::lock_guard<std::mutex> lock(v_mutex);
        v.second = samples;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[17] += elapsed.count();

    }

    void parallel_for3(std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& B, 
                      int rho, std::vector<int>& keys) {
        auto start = std::chrono::high_resolution_clock::now();

        constexpr size_t MAX_THREADS = 20000;
        std::vector<std::future<void>> futures;

        for (auto& v : B) {
            if (futures.size() >= MAX_THREADS) {
                for (auto& fut : futures) {
                    fut.wait();
                }
                futures.clear();
            }
            int v_first = v.first;
            futures.push_back(std::async(std::launch::async, &HNSW::process_entry3, this, std::ref(keys), std::ref(v), v_first));
        }

        for (auto& fut : futures) {
            fut.wait();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[18] += elapsed.count();
    }




    std::unordered_map<int, std::vector<std::tuple<double, int, bool>>> nn_descent_full(
        std::unordered_map<int, std::vector<std::tuple<double, int, bool>>> B, 
        int K, 
        int layer) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        if (B.empty()) {
            return B;
        }

        double delta = 0.001;
        int rho = 5;

        std::vector<int> keys = extract_keys(B);

        parallel_for3(B, rho, keys);

        while (true) {
            parallel_for(B, rho);

            auto oldp = reverse(old);
            auto newp = reverse(new_map);
            int counter = 0;
            
            parallel_for2(B, rho, layer, counter);

            if (counter < delta * B.size() * K) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                this->time[3] += elapsed.count();
                return B;
            }
        }
    }

    std::unordered_map<int, std::set<int>> transform_heaps_to_sets_full(
    const std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& B) 
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_map<int, std::set<int>> transformed_B;

        for (const auto& item : B) {
            int q = item.first;
            const auto& neighbors = item.second;
            std::set<int> neighbor_set;
            
            for (const auto& neighbor_tuple : neighbors) {
                double priority; int neighbor; bool flag;
                std::tie(priority, neighbor, flag) = neighbor_tuple;
                neighbor_set.insert(neighbor);
            }

            transformed_B[q] = neighbor_set;
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[4] += elapsed.count();
        return transformed_B;
    }


    std::vector<std::pair<double, int>> sample(const std::vector<int>& V, int K, int m) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<int> keys_without_m;
        
        for (const auto& key : V) {
            if (key != m) {
                keys_without_m.push_back(key);
            }
        }
        
        std::vector<int> sampled_neighbors;
        if (K <= keys_without_m.size()) {
            std::unordered_set<int> selected_indices;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, keys_without_m.size() - 1);
            
            while (selected_indices.size() < K) {
                selected_indices.insert(dis(gen));
            }
            
            for (int index : selected_indices) {
                sampled_neighbors.push_back(keys_without_m[index]);
            }
        } else {
            sampled_neighbors = keys_without_m;
        }
        
        std::vector<std::pair<double, int>> heap;
        for (const auto& neighbor : sampled_neighbors) {
            heap.emplace_back(-std::numeric_limits<double>::infinity(), neighbor);
        }
        std::make_heap(heap.begin(), heap.end());
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[5] += elapsed.count();
        
        return heap;
    }

    std::unordered_map<int, std::vector<std::pair<double, int>>> reverse(
    const std::unordered_map<int, std::vector<std::pair<double, int>>>& B) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::unordered_map<int, std::vector<std::pair<double, int>>> R;

        for (const auto& pair : B) {
            int u = pair.first;
            const auto& neighbors = pair.second;

            for (const auto& neighbor : neighbors) {
                double neg_distance; int v;
                std::tie(neg_distance, v) = neighbor;
                if (u != v) {
                    R[v].emplace_back(neg_distance, u);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[6] += elapsed.count();

        return R;
    }


    std::vector<int> extract_keys(const std::unordered_map<int, std::vector<std::tuple<double, int, bool>>>& V) {
        std::vector<int> keys;
        keys.reserve(V.size());
        for (const auto& item : V) {
            keys.push_back(item.first);
        }
        return keys;
    }


    std::vector<std::tuple<double, int, bool>> sample_full(
    std::vector<int>& keys,
    int m) 
    {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_set<int> sampled_neighbors;
        std::vector<std::tuple<double, int, bool>> heap;

        if (K < keys.size() - 1) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, keys.size() - 1);

            while (sampled_neighbors.size() < K) {
                int idx = dis(gen);
                if (keys[idx] != m) {
                    heap.emplace_back(std::numeric_limits<double>::infinity(), keys[idx], true);
                    sampled_neighbors.insert(keys[idx]);
                }
            }
        } else {
            for (int elem : keys) {
                if (elem != m) {
                    heap.emplace_back(std::numeric_limits<double>::infinity(), elem, true);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = end - start;

        this->time[7] += elapsed.count();

        return heap;
    }


    std::vector<std::pair<double, int>> sample_full2(
        const std::vector<std::pair<double, int>>& V,
        int K) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<double, int>> keys_without_m = V;

        std::vector<std::pair<double, int>> sampled_neighbors;
        if (K <= keys_without_m.size()) {
            std::unordered_set<int> selected_indices;
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, keys_without_m.size() - 1);

            while (selected_indices.size() < K) {
                selected_indices.insert(dis(gen));
            }

            for (int index : selected_indices) {
                sampled_neighbors.push_back(keys_without_m[index]);
            }
        } else {
            sampled_neighbors = keys_without_m;
        }

        std::vector<std::pair<double, int>> heap;
        for (const auto& neighbor : sampled_neighbors) {
            heap.emplace_back(neighbor.first, neighbor.second);
        }
        std::make_heap(heap.begin(), heap.end());

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[8] += elapsed.count();

        return heap;
    }

    int update_nn_full(
    std::vector<std::tuple<double, int, bool>>& heap,
    const std::tuple<double, int, bool>& neighbor_with_neg_dist,
    int l,
    int v,
    int Mmax) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        double neg_distance;
        int neighbor;
        bool b;
        std::tie(neg_distance, neighbor, b) = neighbor_with_neg_dist;

        bool found = false; 

        for (auto it = heap.begin(); it != heap.end(); ) {
            if (std::get<1>(*it) == neighbor && std::isinf(std::get<0>(*it))) {
                it = heap.erase(it); 
                found = true;
                std::make_heap(heap.begin(), heap.end()); 
                heap.push_back(neighbor_with_neg_dist);
                std::push_heap(heap.begin(), heap.end());
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                this->time[9] += elapsed.count();
                return 1;
            } else {
                ++it;
            }
        }
        
        if (!found) {
            if (neg_distance < std::get<0>(heap.front())) {
                std::pop_heap(heap.begin(), heap.end());
                int removed_neighbor = std::get<1>(heap.back());
                heap.pop_back();
                heap.push_back(std::make_tuple(neg_distance, neighbor, true));
                std::push_heap(heap.begin(), heap.end());

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                this->time[9] += elapsed.count();
                return 1;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[9] += elapsed.count();
        return 0;
    }


    std::vector<std::pair<double, int>> merge_heaps_full(
    const std::vector<std::pair<double, int>>& B,
    const std::vector<std::pair<double, int>>& R) 
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::pair<double, int>> merged_list = B;
        std::make_heap(merged_list.begin(), merged_list.end());

        for (const auto& item : R) {
            merged_list.push_back(item);
            std::push_heap(merged_list.begin(), merged_list.end());
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[10] += elapsed.count();
        return merged_list;
    }

    std::unordered_set<int> neighborhood(int element, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_set<int> result;
        if (lc < top_layer) {
            auto it = layers[lc].find(element);
            if (it != layers[lc].end()) {
                for (const auto& neighbor_tuple : it->second) {
                    int neighbor;
                    std::tie(std::ignore, neighbor, std::ignore) = neighbor_tuple;
                    result.insert(neighbor);
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        this->time[11] += elapsed.count();

        return result;
    }

    std::vector<int> searchLayer(int q, const std::vector<int>& ep, int ef, int lc) {
        auto start = std::chrono::high_resolution_clock::now();

        std::unordered_set<int> visited;
        std::priority_queue<std::pair<double, int>> C; 
        std::priority_queue<std::pair<double, int>> W;

        for (int e : ep) {
            double distance_e_q = distance(e, q);
            C.emplace(-distance_e_q, e);
            W.emplace(distance_e_q, e);
            visited.insert(e);
        }

        while (!C.empty()) {
            auto [c_dist, c_elem] = C.top();
            C.pop();
            if (W.top().first < -c_dist) {
                break;
            }

            for (int e : neighborhood(c_elem, lc)) {
                if (visited.find(e) == visited.end()) {
                    double distance_e_q = distance(e, q);
                    visited.insert(e);

                    if (W.size() < ef || distance_e_q < W.top().first) {
                        if (distance_e_q != 0) {
                            C.emplace(-distance_e_q, e);
                            W.emplace(distance_e_q, e);
                            if (W.size() > ef) {
                                W.pop();
                            }
                        }
                    }
                }
            }
        }

        std::vector<int> result;
        while (!W.empty() && result.size() < ef) {
            result.push_back(W.top().second);
            W.pop();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        return result;
    }



};




namespace py = pybind11;

PYBIND11_MODULE(hnsw_tanimoto, m) {
    py::class_<HNSW>(m, "HNSW")
        .def(py::init<int, int, float, const std::string&>(),
             py::arg("K"), py::arg("efConstruction"), py::arg("mL"), py::arg("filename"))
        .def("insert_list", &HNSW::insert_list, py::arg("indices"))
        .def("k_nn_search", &HNSW::k_nn_search, py::arg("index"), py::arg("K"), py::arg("efConstruction"))
        .def("distance", &HNSW::distance, py::arg("idx1"), py::arg("idx2"))
        .def("print_structure", &HNSW::print_structure)
        .def_readwrite("time", &HNSW::time);
}