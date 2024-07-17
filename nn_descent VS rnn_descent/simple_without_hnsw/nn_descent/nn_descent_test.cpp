#include <vector>
#include <limits>
#include <random>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <thread>
#include <iostream>
#include <vector>
#include <iostream>
#include <vector>
#include <random>
#include <array>

#include <omp.h>

#include <fstream>


namespace nng
{

	template <class DataT>
	struct NNDescent
	{

	public:
		typedef DataT data_type;

		typedef float sim_value_type;

		sim_value_type sim(const data_type &a, const data_type &b){
			float sum = 0.0;
			for (size_t i = 0; i < a.size(); ++i)
			{
				float dx = a[i] - b[i];
				sum += dx * dx;
			}
			return -sqrt(sum);
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

	public:
		NNDescent(const uint32_t K, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0))
			: K(K), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nodes(), nbds(), rand(rand), checked()
		{
		}

		NNDescent(const std::size_t K, const std::vector<data_type> &datas, const double rho = 0.5, const double perturb_rate = 0, const std::size_t num_random_join = 10, const rand_type &rand = rand_type(0))
			: K(K), nodes(), num_random_join(num_random_join), rho(rho), perturb_rate(perturb_rate), nbds(), rand(rand), checked()
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

	public:
		uint32_t K;
		uint32_t num_random_join; // random join size
		double rho;				  // sample rate
		double perturb_rate;
		std::vector<node_type> nodes;
		std::vector<nbd_type> nbds;
		rand_type rand;

	private:
		std::vector<std::vector<bool>> checked;

	}; // end of struct NNDescent

}; // end of nng



typedef std::vector<float> PointND;

struct EuclideanDistance
{
	typedef float value_type;
	float operator()(const PointND &a, const PointND &b) const
	{
		float sum = 0.0;
		for (size_t i = 0; i < a.size(); ++i)
		{
			float dx = a[i] - b[i];
			sum += dx * dx;
		}
		return sqrt(sum);
	}
};

struct EuclideanSim
{
	typedef float value_type;
	float operator()(const PointND &a, const PointND &b) const
	{
		float sum = 0.0;
		for (size_t i = 0; i < a.size(); ++i)
		{
			float dx = a[i] - b[i];
			sum += dx * dx;
		}
		return -sqrt(sum);
	}
};

int main()
{
    const int N = 10000;
    const int K = 20;
    const int T = 20;
    const int L = 20;  // Number of nearest neighbors to check

    std::vector<int> dimensions = {10, 20, 30, 40, 50, 70, 100, 120, 150, 180, 200, 250, 300, 500, 800, 1000};
    std::vector<double> execution_times;
    std::vector<double> correct_percentages;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1000);

    for (int DIM : dimensions)
    {
        std::vector<PointND> points;

        for (int i = 0; i < N; ++i)
        {
            PointND point(DIM);
            for (int d = 0; d < DIM; ++d)
            {
                point[d] = dis(gen);
            }
            points.push_back(point);
        }

        nng::NNDescent<PointND> nn(K);
        nn.init_graph(points);

        auto start = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < T; t++)
        {
            nn.update();
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        execution_times.push_back(elapsed.count());

        std::cout << "Dimension: " << DIM << " - Time for NNDescent updates: " << elapsed.count() << " seconds" << std::endl;

        // Evaluate the correctness of NNDescent
        int index = rd() % N;
        const auto &target = points[index];

        std::vector<std::pair<int, float>> naive_neighbors;
        for (int i = 0; i < N; ++i)
        {
            if (i == index)
                continue;
            float dist = EuclideanDistance()(target, points[i]);
            naive_neighbors.push_back({i, dist});
        }
        std::sort(naive_neighbors.begin(), naive_neighbors.end(),
                  [](const std::pair<int, float> &a, const std::pair<int, float> &b)
                  {
                      return a.second < b.second;
                  });

        std::unordered_set<int> nnDescentNeighbors;
        for (const auto &neighbor : nn.nbds[index])
        {
            nnDescentNeighbors.insert(neighbor.node_id);
        }

        int correctCount = 0;
        for (int i = 0; i < L; ++i)
        {
            if (nnDescentNeighbors.find(naive_neighbors[i].first) != nnDescentNeighbors.end())
            {
                correctCount++;
            }
        }

        double percentageCorrect = (double)correctCount / L * 100;
        correct_percentages.push_back(percentageCorrect);
        std::cout << "Percentage of correct neighbors found by NNDescent: " << percentageCorrect << "%" << std::endl;
    }

    // Write results to a file for plotting
    std::ofstream file("nn_descent_results.csv");
    file << "Dimension,ExecutionTime,CorrectPercentage\n";
    for (size_t i = 0; i < dimensions.size(); ++i)
    {
        file << dimensions[i] << "," << execution_times[i] << "," << correct_percentages[i] << "\n";
    }
    file.close();

    std::cout << "Results written to nn_descent_results.csv" << std::endl;

    return 0;
}
