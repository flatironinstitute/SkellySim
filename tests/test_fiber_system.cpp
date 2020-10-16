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

    const int n_pts = 48;
    const int n_fib_per_rank = 3000 / size;

    FiberContainer fibs(n_fib_per_rank, n_pts, 0.1);
    Eigen::MatrixXd f_fib = Eigen::MatrixXd::Zero(3, n_pts * n_fib_per_rank);

    for (int i = 0; i < n_fib_per_rank; ++i) {
        fibs.fibers[i].translate(
            {0., 0., 100 * static_cast<double>(i + n_fib_per_rank * rank) / (size * n_fib_per_rank + 1)});
        for (int j = 0; j < n_pts; ++j)
            f_fib(2, i * n_pts + j) = 1.0;
        fibs.fibers[i].length = 1.0;
    }

    double st = omp_get_wtime();
    fibs.update_stokeslets();
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    auto vel = fibs.flow(f_fib);
    st = omp_get_wtime();
    vel = fibs.flow(f_fib);
    if (rank == 0)
        std::cout << omp_get_wtime() - st << std::endl;

    if (rank == 0)
        std::cout << "Test passed\n";

    MPI_Finalize();

    return 0;
}
