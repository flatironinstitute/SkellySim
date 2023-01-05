#include <rng.hpp>

#include <trng/normal_dist.hpp>
#include <trng/poisson_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/uniform_int_dist.hpp>
#include <trng/yarn2.hpp>

#include <iostream>

#include <mpi.h>

namespace RNG {
using engine_type = trng::yarn2;
engine_type engine_distributed;
engine_type engine_shared;

void init(unsigned long seed) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    engine_shared = engine_type();
    engine_distributed = engine_type();

    engine_distributed.seed(seed);
    engine_shared.seed(seed);

    engine_shared.split(2, 0);

    engine_distributed.split(2, 1);
    engine_distributed.split(size, rank);
}

void init(std::pair<std::string, std::string> state) {
    std::stringstream shared_stream;
    std::stringstream distributed_stream;

    shared_stream.str(state.first);
    distributed_stream.str(state.second);

    shared_stream >> engine_shared;
    distributed_stream >> engine_distributed;
}

std::pair<std::string, std::string> dump_state() {
    std::stringstream shared_stream;
    std::stringstream distributed_stream;

    shared_stream << engine_shared;
    distributed_stream << engine_distributed;
    return std::make_pair(shared_stream.str(), distributed_stream.str());
}

double uniform(double low, double high) { return trng::uniform_dist(low, high)(engine_distributed); }
int uniform_int(int low, int high) { return trng::uniform_int_dist(low, high)(engine_distributed); }
double normal(double mu, double sigma) { return trng::normal_dist(mu, sigma)(engine_distributed); }
int poisson_int(double mu) { return trng::poisson_dist(mu)(engine_distributed); }

double uniform_unsplit(double low, double high) { return trng::uniform_dist(low, high)(engine_shared); }
int uniform_int_unsplit(int low, int high) { return trng::uniform_int_dist(low, high)(engine_shared); }
double normal_unsplit(double mu, double sigma) { return trng::normal_dist(mu, sigma)(engine_shared); }
int poisson_int_unsplit(double mu) { return trng::poisson_dist(mu)(engine_shared); }
} // namespace RNG
