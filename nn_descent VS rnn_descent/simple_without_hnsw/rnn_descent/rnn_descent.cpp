#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <cstring>
#include <mutex>
#include <cmath>
#include <omp.h>
#include <set>

int ntotal = 10000;
int L = 48;
int S = 20;
int random_seed = 42;
int T1 = 4;
int T2 = 15;

class DistanceComputer {
public:
    const std::vector<std::vector<double>>& points;

    DistanceComputer(const std::vector<std::vector<double>>& points) : points(points) {}

    double symmetric_dis(int i, int j) {
        if (i == j) {
            return 0.0f;
        }
        if (i >= points.size() || j >= points.size()) {
            std::cerr << "Index out of bounds" << std::endl;
            return -1.0f; 
        }
        return compute_distance(points[i], points[j]);
    }

    double operator()(int i) {
        if (i >= points.size()) {
            std::cerr << "Index out of bounds" << std::endl;
            return -1.0f;
        }
        return compute_distance(points[i], points[0]);
    }

    static double compute_distance(const std::vector<double>& p1, const std::vector<double>& p2) {
        double sum = 0.0f;
        for (size_t k = 0; k < p1.size(); ++k) {
            double diff = p1[k] - p2[k];
            sum += diff * diff;
        }
        return std::sqrt(sum);
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
    int d;
    int ntotal;
    int L, S;
    int random_seed;
    bool has_built;
    std::vector<int> offsets;
    std::vector<int> final_graph;

    RNNDescent(const int d, int L, int S, int random_seed)
        : d(d), L(L), S(S), random_seed(random_seed), has_built(false) {}

    ~RNNDescent() {}

    void init_graph(DistanceComputer& qdis) {
        graph.reserve(ntotal);
        {
            std::mt19937 rng(random_seed * 6007);
            for (int i = 0; i < ntotal; i++) {
                graph.push_back(Nhood(L, S, rng, (int)ntotal));
            }
        }

        #pragma omp parallel
        {
            std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
            #pragma omp for
            for (int i = 0; i < ntotal; i++) {
                std::vector<int> tmp(S);

                Nhood::gen_random(rng, tmp.data(), S, ntotal);

                for (int j = 0; j < S; j++) {
                    int id = tmp[j];
                    if (id == i) continue;
                    double dist = qdis.symmetric_dis(i, id);

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

    void update_neighbors(DistanceComputer& qdis) {
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
                    double distance = qdis.symmetric_dis(nn.id, other_nn.id);
                    if (distance < nn.distance) {
                        ok = false;
                        insert_nn(other_nn.id, nn.id, distance, true);
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

    void build(DistanceComputer& qdis, const int n, bool verbose) {
        if (verbose) {
            printf("Parameters: S=%d, L=%d\n", S, L);
        }

        ntotal = n;
        init_graph(qdis);

        for (int t1 = 0; t1 < T1; ++t1) {
            if (verbose) {
                std::cout << "Iter " << t1 << " : " << std::flush;
            }
            for (int t2 = 0; t2 < T2; ++t2) {
                update_neighbors(qdis);
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

    void search(DistanceComputer& qdis, const int topk, int* indices, double* dists, VisitedTable& vt) const {
        if (!has_built) {
            throw std::runtime_error("The index is not built yet.");
        }
        int L = std::max(this->L, topk);

        std::vector<Neighbor> retset(L + 1);

        std::vector<int> init_ids(L);
        std::mt19937 rng(random_seed);

        Nhood::gen_random(rng, init_ids.data(), L, ntotal);
        for (int i = 0; i < L; i++) {
            int id = init_ids[i];
            double dist = qdis(id);
            retset[i] = Neighbor(id, dist, true);
        }

        std::sort(retset.begin(), retset.begin() + L);

        int k = 0;

        while (k < L) {
            int nk = L;

            if (retset[k].flag) {
                retset[k].flag = false;
                int n = retset[k].id;

                int offset = offsets[n];
                int K = std::min(L, offsets[n + 1] - offset);
                for (int m = 0; m < K; ++m) {
                    int id = final_graph[offset + m];
                    if (vt.get(id)) continue;

                    vt.set(id);
                    double dist = qdis.symmetric_dis(n, id);
                    if (dist >= retset[L - 1].distance) continue;

                    Neighbor nn(id, dist, true);
                    //int r = insert_into_pool(retset.data(), L, nn);
                    int r = 2; // pour enlever l'erreur
                    if (r < nk) nk = r;
                }
            }
            if (nk <= k)
                k = nk;
            else
                ++k;
        }
        for (size_t i = 0; i < topk; i++) {
            indices[i] = retset[i].id;
            dists[i] = retset[i].distance;
        }

        vt.advance();
    }

    void reset() {
        has_built = false;
        ntotal = 0;
        final_graph.resize(0);
        offsets.resize(0);
    }

    int insert_into_pool(Neighbor* addr, int size, Neighbor nn) {
        int left = 0, right = size - 1;
        if (addr[left].distance > nn.distance) {
            memmove((char*)&addr[left + 1], &addr[left], size * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if (addr[right].distance < nn.distance) {
            addr[size] = nn;
            return size;
        }
        while (left < right - 1) {
            int mid = (left + right) / 2;
            if (addr[mid].distance > nn.distance)
                right = mid;
            else
                left = mid;
        }
        while (left > 0) {
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return size + 1;
            left--;
        }
        if (addr[left].id == nn.id || addr[right].id == nn.id) return size + 1;
        memmove((char*)&addr[right + 1], &addr[right], (size - right) * sizeof(Neighbor));
        addr[right] = nn;
        return right;
    }
};

int main() {
    int d = 48;
    std::vector<std::vector<double>> points(ntotal, std::vector<double>(d));
    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (auto& point : points) {
        for (auto& coord : point) {
            coord = dist(rng);
        }
    }

    DistanceComputer dc(points);

    RNNDescent rnn(d, L, S, random_seed);

    auto start = std::chrono::high_resolution_clock::now();
    rnn.build(dc, ntotal, true);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> build_duration = end - start;
    std::cout << "Graph construction time: " << build_duration.count() << " seconds." << std::endl;

    std::uniform_int_distribution<int> node_dist(0, ntotal - 1);
    const int random_node_index = 1;
    
    std::vector<std::pair<double, int>> distances;
    for (int i = 0; i < ntotal; ++i) {
        if (i != random_node_index) {
            double dist = dc.symmetric_dis(random_node_index, i);
            distances.push_back({dist, i});
        }
    }
    
    std::sort(distances.begin(), distances.end());

    std::sort(distances.begin(), distances.end());
    std::set<int> closest_neighbors;
    std::cout << "5 closest neighbors of node " << random_node_index << " are: ";
    for (int i = 0; i < 5; ++i) {
        closest_neighbors.insert(distances[i].second);
        std::cout << distances[i].second << " (distance: " << distances[i].first << "), ";
    }
    std::cout << std::endl;

    std::set<int> stored_neighbors;
    std::cout << "Stored neighbors in the graph for node " << random_node_index << ": ";
    for (const auto& neighbor : rnn.graph[random_node_index].pool) {
        stored_neighbors.insert(neighbor.id);
        std::cout << neighbor.id << " (distance: " << neighbor.distance << "), ";
    }
    std::cout << std::endl;

    std::vector<int> common_neighbors;
    std::set_intersection(closest_neighbors.begin(), closest_neighbors.end(),
                          stored_neighbors.begin(), stored_neighbors.end(),
                          std::back_inserter(common_neighbors));

    double percentage = 100.0 * common_neighbors.size() / 5;
    std::cout << "Percentage of true neighbors among calculated ones: " << percentage << "%" << std::endl;

    return 0;
}
