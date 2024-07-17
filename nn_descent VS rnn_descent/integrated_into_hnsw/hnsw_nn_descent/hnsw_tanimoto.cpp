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




namespace nng
{

	template <class DataT>
	struct NNDescent
	{

	public:
        uint32_t K;
		uint32_t num_random_join; // random join size
		double rho;				  // sample rate
		double perturb_rate;
		double distance_time = 0.0;

		typedef DataT data_type;

		typedef float sim_value_type;

		sim_value_type sim(const data_type &a, const data_type &b){
			// auto start_distance = std::chrono::high_resolution_clock::now();

			const std::vector<char>& new_fp1 = new_fps[a];
            const std::vector<char>& new_fp2 = new_fps[b];

            int common_indices = 0;
            int common_negative = 0;

            for (size_t i = 0; i < new_fp1.size(); ++i) {
                int fp1_val = new_fp1[i], fp2_val = new_fp2[i];
                bool valid_indices = (fp1_val != -1 && fp2_val != -1);
                common_indices += (valid_indices && fp1_val == fp2_val);
                common_negative += (fp1_val == -1 && fp2_val == -1);
            }

            int adjusted_fps_bits = new_fp1.size() - common_negative;

			// auto end_distance = std::chrono::high_resolution_clock::now();
			// std::chrono::duration<double> elapsed_distance = end_distance - start_distance;
			// distance_time += elapsed_distance.count();

            return adjusted_fps_bits ? static_cast<double>(common_indices) / adjusted_fps_bits : 0.0;
		}

		struct node_type
		{
			node_type(const data_type &data) : data(data) {}
			node_type() : data() {}
			data_type data;
		};

		struct nbd_elem_type
		{
			nbd_elem_type() : node_id(), similarity(0.0) {}
			nbd_elem_type(const std::size_t node_id, const sim_value_type similarity) : node_id(node_id), similarity(similarity) {}
			std::size_t node_id;
			sim_value_type similarity;
		};

		struct comp_similarity
		{
			inline bool operator()(const nbd_elem_type &e1, const nbd_elem_type &e2) const
			{
				return e1.similarity > e2.similarity;
			}
		};

		typedef std::vector<nbd_elem_type> nbd_type;

		struct nearer
		{
			inline bool operator()(const nbd_elem_type &elem1, const nbd_elem_type &elem2) const
			{
				return elem1.similarity > elem2.similarity;
			}
		};

		typedef std::mt19937 rand_type;

        std::vector<node_type> nodes;
		std::vector<nbd_type> nbds;
		rand_type rand;

	public:
		NNDescent(const uint32_t K, std::vector<std::vector<char>> new_fps, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0))
			: K(K), new_fps(new_fps), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nodes(), nbds(), rand(rand), checked()
		{
		}

		NNDescent(const std::size_t K, std::vector<std::vector<char>> new_fps, const std::vector<data_type> &datas, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0))
			: K(K), new_fps(new_fps), nodes(), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nbds(), rand(rand), checked()
		{
			init_graph(datas);
		}

	public:
		void init_graph(const std::vector<data_type> &datas)
		{
			if (datas.size() == 0)
			{
				return;
			}

			// clear
			this->clear();

			// set datas
			for (const data_type &data : datas)
			{
				this->nodes.push_back(node_type(data));
			}

			// construct random neighborhoods
			this->init_random_nbds();
		}

		std::size_t update()
		{
			if (this->nodes.size() < 2 || this->K == 0)
			{
				return 0;
			} // cannot create nbd

			const std::size_t N = this->nodes.size();

			std::vector<nbd_type> old_nbds(N), new_nbds(N), old_rnbds(N), new_rnbds(N);

// Process 1
// set old_nbds / new_nbds
#pragma omp parallel for
			for (std::size_t i = 0; i < N; ++i)
			{ // ToDo: parallel for i
				this->prep_nbd(i, new_nbds[i], old_nbds[i]);
			}

			// Process 2
			// set old_rnbds / new_rnbds
			for (std::size_t i = 0; i < N; ++i)
			{
				const auto &new_nbd(new_nbds[i]), &old_nbd(old_nbds[i]);
				for (const nbd_elem_type &elem : old_nbd)
				{
					assert(elem.node_id != i);
					old_rnbds[elem.node_id].push_back(nbd_elem_type(i, elem.similarity));
				}
				for (const nbd_elem_type &elem : new_nbd)
				{
					assert(elem.node_id != i);
					new_rnbds[elem.node_id].push_back(nbd_elem_type(i, elem.similarity));
				}
			}

			// Process 3
			// local join
			std::size_t update_count = 0;
#pragma omp parallel for
			for (std::size_t i = 0; i < N; ++i)
			{ // ToDo: parallel for i
				update_count += this->local_join(i, new_nbds[i], old_nbds[i], new_rnbds[i], old_rnbds[i]);
			}

			return update_count;
		}

		void operator()(const std::vector<data_type> &datas, const std::size_t &max_epoch = 100, const double &delta = 0.001)
		{
			this->operator()(this->K, datas, max_epoch, delta);
		}

		void operator()(const std::size_t K, const std::vector<data_type> &datas, const std::size_t &max_epoch = 100, const double &delta = 0.001)
		{
			this->K = K;

			// initialize graph
			std::cout << "init graph ... " << std::flush;
			this->init_graph(datas);
			std::cout << "done" << std::endl;

			// Update
			for (std::size_t epoch = 0; epoch < max_epoch; ++epoch)
			{
				std::cout << "update [" << epoch + 1 << "/" << max_epoch << "] ..." << std::flush;
				std::size_t update_count = update();
				std::size_t KN = this->rho * this->K * this->nodes.size();
				std::cout << " " << update_count << "/" << KN << std::endl;
				if (update_count <= delta * KN)
				{
					break;
				}
			}
		}

	private:
        std::vector<std::vector<char>> new_fps;
		void clear()
		{
			this->nodes.clear();
			this->nbds.clear();
			this->checked.clear();
		}

		void init_random_nbds()
		{
			const auto N = this->nodes.size();

			// const uint32_t K2 = (K <= N-1)? K : static_cast<uint32_t>(N-1);
			uint32_t K2 = this->K;
			if (K2 > N - 1)
			{
				K2 = N - 1;
			}
			assert(K2 <= this->K && K2 <= N - 1);
			this->nbds.clear();
			this->nbds.resize(N);
			this->checked.clear();
			this->checked.resize(N);

			if (N < 2 || this->K == 0)
			{
				return;
			} // cannot create nbd
#pragma omp parallel for
			for (std::size_t i = 0; i < N; ++i)
			{ // ToDo: parallel for i

				std::unordered_set<std::size_t> chosen;

				// assumed K << N
				for (std::size_t j = 0; j < K2; ++j)
				{
					std::size_t n = rand() % (N - 1);
					if (n >= i)
					{
						++n;
					}
					assert(i != n);
					chosen.insert(n);
				}
				assert(chosen.size() <= K2);
				// set neighborhood
				const auto &node(this->nodes[i]);
				auto &nbd(this->nbds[i]);
				nbd.resize(chosen.size());
				std::size_t count = 0;
				for (auto j : chosen)
				{
					assert(i != j);
					nbd[count++] = nbd_elem_type(j, this->sim(node.data, this->nodes[j].data));
				}
				std::sort(nbd.begin(), nbd.end(), nearer());

				this->checked[i].resize(nbd.size(), false);

				assert(nbd.size() > 0); // because K > 0 and N > 1
				assert(nbd.size() <= this->K);
			}
		}

		typedef enum
		{
			NOT_INSERTED,
			INSERTED
		} join_result_type;

		std::size_t compute_ub(const nbd_type &nbd, const std::size_t joiner,
							   const sim_value_type s, const std::size_t K2) const
		{
			assert(K2 > 0);
			if (nbd.back().similarity >= s)
			{
				return K2;
			}
			else
			{
				return std::upper_bound(nbd.begin(), nbd.end(), nbd_elem_type(joiner, s), comp_similarity()) - nbd.begin();
			}
		}

		join_result_type join(const std::size_t base, const std::size_t joiner)
		{
			assert(base != joiner);
			assert(this->nodes.size() > 1 && this->K > 0);
			const auto &base_node(this->nodes[base]), &joiner_node(this->nodes[joiner]);
			auto &nbd(this->nbds[base]);
			auto &chkd(this->checked[base]);
			assert(nbd.size() == chkd.size());
			const auto s = this->sim(base_node.data, joiner_node.data);
			const auto K2 = this->nbds[base].size();

			if (s < nbd.back().similarity && K2 == this->K)
			{
				return NOT_INSERTED;
			}

			const nbd_elem_type joiner_elem(joiner, s);
			assert(K2 > 0);
			assert(K2 <= this->K && K2 <= this->nodes.size() - 1);

			// find the position i such that nbd[i] is where joiner will be inserted

			// search ub
			const std::size_t ub = std::upper_bound(nbd.begin(), nbd.end(), joiner_elem, comp_similarity()) - nbd.begin();

			// to prevent perturbation of nbd, the probability of replacement
			// with most dissimilar element in nbd shoule be suppressed
			//      if(ub == K2 && (rand() % B) >= prB){
			//      const auto B = 1000000; // Big Integer (ad-hoc)
			const auto SHIFT = 20;
			const auto B = 1 << SHIFT; // Big Integer (ad-hoc)
			const auto prB = this->perturb_rate * B;
			if (ub == K2 && nbd.back().similarity == s && (rand() & (B - 1)) >= prB)
			{ // ub == K2 && rand() <= perturb_rate
				return NOT_INSERTED;
			}

			// search lb
			const std::size_t lb = std::lower_bound(nbd.begin(), nbd.begin() + ub, joiner_elem, comp_similarity()) - nbd.begin();

			if (K2 > 0 && nbd[lb].similarity == s)
			{
				for (std::size_t i = lb; i < ub; ++i)
				{
					if (nbd[i].node_id == joiner)
					{
						return NOT_INSERTED;
					} // joiner still in nbd
				}
			}

			assert(lb <= ub);
			auto pos = (lb < ub) ? lb + rand() % (ub - lb) : lb;

			// insert
			if (K2 < this->K)
			{
				nbd.insert(nbd.begin() + pos, joiner_elem);
				chkd.insert(chkd.begin() + pos, false);
			}
			else
			{
				assert(K2 == this->K);
				nbd_elem_type cur_elem(joiner_elem);
				bool cur_checked = false;
				for (std::size_t i = pos; i < K2; ++i)
				{
					std::swap(cur_elem, nbd[i]);
					std::swap(cur_checked, chkd[i]);
				}
			}

			return INSERTED;
		}

		void prep_nbd(const std::size_t i, nbd_type &new_nbd, nbd_type &old_nbd)
		{
			const std::size_t N = this->nodes.size();
			const nbd_type &nbd(this->nbds[i]);
			const std::size_t K2 = nbd.size();
			const std::size_t rhoK = std::min(static_cast<std::size_t>(std::ceil(this->rho * this->K)), N - 1);

			std::vector<std::size_t> sampled;
			for (std::size_t j = 0, n = 0; j < K2; ++j)
			{
				assert(nbd[j].node_id != i);
				if (this->checked[i][j])
				{
					old_nbd.push_back(nbd[j]);
				}
				else
				{
					// choose rhoK unchecked element with reservoir sampling
					if (n < rhoK)
					{
						sampled.push_back(j);
					}
					else
					{
						std::size_t m = rand() % (n + 1);
						if (m < rhoK)
						{
							sampled[m] = j;
						}
					}
					++n;
				}
			}

			for (const std::size_t j : sampled)
			{
				assert(i != nbd[j].node_id);
				this->checked[i][j] = true;
				new_nbd.push_back(nbd[j]);
			}
		}

		std::size_t local_join(const std::size_t i, nbd_type &new_nbd, nbd_type &old_nbd,
							   const nbd_type &new_rnbd, const nbd_type &old_rnbd)
		{

			std::size_t update_count = 0;
			const std::size_t N = this->nodes.size();
			const std::size_t rhoK = std::min(static_cast<std::size_t>(std::floor(this->rho * this->K)), N - 1);

			// old_nbd = old_nbd \cup sample(old_rnbd, rhoK)
			for (const auto &elem : old_rnbd)
			{
				if (rand() % old_rnbd.size() < rhoK)
				{
					old_nbd.push_back(elem);
				}
			}

			// new_nbd = new_nbd \cup sample(new_rnbd, rhoK)
			for (const auto &elem : new_rnbd)
			{
				if (rand() % new_rnbd.size() < rhoK)
				{
					new_nbd.push_back(elem);
				}
			}

			// join(new_nbd, this)
			for (const auto &elem : new_nbd)
			{
				assert(elem.node_id != i);
				update_count += this->join(elem.node_id, i);
			}

			// join(new_nbd, new_ndb)
			for (std::size_t j1 = 0, M = new_nbd.size(); j1 < M; ++j1)
			{
				const auto &elem1(new_nbd[j1]);
				for (std::size_t j2 = j1 + 1; j2 < M; ++j2)
				{
					const auto &elem2(new_nbd[j2]);
					if (elem1.node_id == elem2.node_id)
					{
						continue;
					}
					update_count += this->join(elem1.node_id, elem2.node_id);
					update_count += this->join(elem2.node_id, elem1.node_id);
				}
			}

			// join(new_nbd, old_ndb)
			for (const auto &elem1 : new_nbd)
			{
				for (const auto &elem2 : old_nbd)
				{
					if (elem1.node_id == elem2.node_id)
					{
						continue;
					}
					update_count += this->join(elem1.node_id, elem2.node_id);
					update_count += this->join(elem2.node_id, elem1.node_id);
				}
			}

			// random_join
			for (std::size_t j = 0; j < this->num_random_join; ++j)
			{
				std::size_t node_id = rand() % (this->nodes.size() - 1);
				if (node_id >= i)
				{
					++node_id;
				}
				this->join(i, node_id);
			}

			return update_count;
		}
		

	private:
		std::vector<std::vector<bool>> checked;

	}; // end of struct NNDescent

}; // end of nng



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

            const int K = 20;
	        const int T = 20;

            nng::NNDescent<int> nn(K, new_fps);
			auto start_build = std::chrono::high_resolution_clock::now();
			
            nn.init_graph(keys);

            auto start = std::chrono::high_resolution_clock::now();
            for (int t = 0; t < T; t++)
            {
                nn.update();
            }

			auto end_build = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_build = end_build - start_build;
			this->time[3] += elapsed_build.count();

			this->time[2] += nn.distance_time;


			auto start_save = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < keys.size(); ++i) {
                std::unordered_set<int> neigh = {};
                for (const auto &neighbor : nn.nbds[i]) {
                    neigh.insert(keys[neighbor.node_id]);
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
            double distance_e_q = new_fps_distance(e, q);
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
                    double distance_e_q = new_fps_distance(e, q);
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
