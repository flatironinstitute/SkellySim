#include <rng.hpp>

#include <trng/normal_dist.hpp>
#include <trng/uniform_dist.hpp>
#include <trng/yarn2.hpp>

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

double uniform(double low, double high) { return trng::uniform_dist(low, high)(engine_distributed); }
double normal(double mu, double sigma) { return trng::normal_dist(mu, sigma)(engine_distributed); }

double uniform_unsplit(double low, double high) { return trng::uniform_dist(low, high)(engine_shared); }
double normal_unsplit(double mu, double sigma) { return trng::normal_dist(mu, sigma)(engine_shared); }

} // namespace RNG
