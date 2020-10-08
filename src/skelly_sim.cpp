#include <Eigen/Dense>
#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Get the number of processes
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    std::cout << "Rank " << rank << " has " << omp_get_max_threads() << " available OpenMP threads on "
              << processor_name << ".\n";

    // Finalize the MPI environment.
    MPI_Finalize();

    return 0;
}
