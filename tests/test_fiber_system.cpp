#include <fstream>
#include <iostream>
#include <mpi.h>

#include <fiber.hpp>
#include <omp.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef NDEBUG
#undef NDEBUG
#include <cassert>
#define NDEBUG
#else
#include <cassert>
#endif

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    spdlog::stdout_color_mt("STKFMM");
    spdlog::stdout_color_mt("Belos");
    spdlog::stdout_color_mt("global-status");

    const int n_pts = 32;
    const int n_time = 1;
    const double eta = 1.0;
    const double length = 1.0;
    const double bending_rigidity = 10.0;
    const double dt = 0.005;
    const std::string input_file = "2K_MTs_onCortex_R5_L1.toml";

    toml::value config = toml::parse(input_file);
    Params params(config.at("params"));
    FiberContainer fibs(config.at("fibers").as_array(), params);

    Eigen::MatrixXd f_fib = Eigen::MatrixXd::Zero(3, fibs.get_local_node_count());

    for (auto &fib : fibs.fibers) {
        assert(fib.length_ == length);
        assert(fib.bending_rigidity_ == bending_rigidity);
        assert(fib.n_nodes_ == n_pts);
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
