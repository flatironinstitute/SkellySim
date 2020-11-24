#include <cnpy.hpp>
#include <periphery.hpp>

#include <mpi.h>

Periphery::Periphery(const std::string &precompute_file) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cnpy::npz_t all_data;

    if (world_rank == 0)
        std::cout << "Loading raw precomputation data from file " << precompute_file << " for periphery into rank 0\n";
    int nrows;
    if (world_rank == 0) {
        all_data = cnpy::npz_load(precompute_file);
        nrows = all_data.at("quadrature_weights").shape[0] * 3;
    }

    MPI_Bcast((void *)&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int ncols = nrows;
    const int row_size_big = nrows / world_size + 1;
    const int row_size_small = nrows / world_size;
    const int nrows_local = (nrows % world_size > world_rank) ? row_size_big : row_size_small;

    // TODO: prevent overflow for large matrices in periphery import
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);
    if (world_rank == 0) {
        int n_rows_big = nrows % world_size;
        for (int i = 0; i < world_size; ++i)
            counts[i] = ncols * ((i < n_rows_big) ? row_size_big : row_size_small);

        for (int i = 1; i < world_size; ++i)
            displs[i] = displs[i - 1] + counts[i - 1];
    }
    const double *M_inv_raw = (world_rank == 0) ? all_data["M_inv"].data<double>() : NULL;
    const double *stresslet_plus_complementary_raw =
        (world_rank == 0) ? all_data["stresslet_plus_complementary"].data<double>() : NULL;

    M_inv_.resize(ncols, nrows_local);
    MPI_Scatterv(M_inv_raw, counts.data(), displs.data(), MPI_DOUBLE, M_inv_.data(), nrows_local * ncols, MPI_DOUBLE, 0,
                 MPI_COMM_WORLD);

    stresslet_plus_complementary_.resize(ncols, nrows_local);
    MPI_Scatterv(M_inv_raw, counts.data(), displs.data(), MPI_DOUBLE, stresslet_plus_complementary_.data(),
                 nrows_local * ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    all_data.clear();

    M_inv_ = M_inv_.transpose();
    stresslet_plus_complementary_ = stresslet_plus_complementary_.transpose();

    std::cout << world_rank << " " << M_inv_.size() << std::endl;

    if (world_rank == 0)
        std::cout << "Done initializing periphery\n";
}
