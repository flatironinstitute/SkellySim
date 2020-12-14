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
    const double bending_rigidity = 0.1;
    const double dt = 0.005;
    const double f_stall = 1.0;
    const std::string fiber_file = "2K_MTs_onCortex_R5_L1.fibers";

    FiberContainer fibs(fiber_file, f_stall, eta);
    Eigen::MatrixXd f_fib = Eigen::MatrixXd::Zero(3, fibs.get_total_fib_points());

    for (auto &fib : fibs.fibers) {
        assert(fib.length_ == 1.0);
        assert(fib.bending_rigidity_ == 10.0);
        assert(fib.num_points_ == 32);
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
        fibs.form_linear_operators(dt, eta);
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    if (rank == 0)
        std::cout << "Test passed\n";

    MPI_Finalize();

    return 0;
}
