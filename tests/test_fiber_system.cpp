#include <fstream>
#include <iostream>
#include <mpi.h>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

#include <fiber.hpp>

#include <omp.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int n_pts = 32;
    const int n_time = 1;
    const double eta = 1.0;
    const double length = 1.0;
    const double bending_rigidity = 10.0;
    const double dt = 0.005;
    const double f_stall = 1.0;
    const std::string input_file = "2K_MTs_onCortex_R5_L1.toml";

    toml::table config = toml::parse_file(input_file);
    Params params(config.get_as<toml::table>("params"));
    FiberContainer fibs(config.get_as<toml::array>("fibers"), params);

    Eigen::MatrixXd f_fib = Eigen::MatrixXd::Zero(3, fibs.get_local_node_count());

    for (auto &fib : fibs.fibers) {
        assert(fib.length_ == length);
        assert(fib.bending_rigidity_ == bending_rigidity);
        assert(fib.num_points_ == n_pts);
    }

    double st = omp_get_wtime();
    for (int i = 0; i < n_time; ++i)
        fibs.update_stokeslets(eta);
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    Eigen::MatrixXd r_trg_external;
    auto vel = fibs.flow(f_fib, r_trg_external, eta);
    st = omp_get_wtime();
    for (int i = 0; i < n_time; ++i)
        vel = fibs.flow(f_fib, r_trg_external, eta);
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    st = omp_get_wtime();
    for (int i = 0; i < n_time; ++i)
        fibs.update_linear_operators(dt, eta);
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    if (rank == 0)
        std::cout << "Test passed\n";

    MPI_Finalize();

    return 0;
}
