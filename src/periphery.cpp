#include <cnpy.hpp>
#include <periphery.hpp>

#include <mpi.h>

Periphery::Periphery(const std::string &precompute_file) {
    using Teuchos::RCP;
    using Teuchos::rcp;
    typedef Tpetra::Map<> map_type;
    typedef Tpetra::MultiVector<> matrix_type;

    const Tpetra::Map<>::global_ordinal_type indexBase = 0;

    RCP<const Teuchos::Comm<int>> comm = Tpetra::getDefaultComm();
    const int rank = comm->getRank();

    cnpy::npz_t all_data;

    if (rank == 0)
        std::cout << "Loading raw precomputation data from file " << precompute_file << " for periphery into rank 0\n";
    int nrows;
    if (rank == 0) {
        all_data = cnpy::npz_load(precompute_file);
        nrows = all_data.at("quadrature_weights_periphery").shape[0] * 3;
    }

    MPI_Bcast((void *)&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::cout << "Copying raw data into Tpetra object on rank 0\n";

    RCP<const map_type> rank_0_map;
    const size_t numLocalIndices = (rank == 0) ? nrows : 0;
    rank_0_map = rcp(new map_type(nrows, numLocalIndices, indexBase, comm));
    RCP<matrix_type> rank_0_M_inv(new matrix_type(rank_0_map, nrows));
    RCP<matrix_type> rank_0_stresslet(new matrix_type(rank_0_map, nrows));
    const int ncols = nrows;
    for (auto i_row : rank_0_map->getNodeElementList()) {
        double *M_inv_row = all_data.at("M_inv_periphery").data<double>() + i_row * ncols;
        double *stresslet_row = all_data.at("shell_stresslet").data<double>() + i_row * ncols;
        for (int i_col = 0; i_col < ncols; ++i_col) {
            rank_0_M_inv->replaceLocalValue(i_col, i_row, M_inv_row[i_col]);
            rank_0_stresslet->replaceLocalValue(i_col, i_row, stresslet_row[i_col]);
        }
    }
    all_data.clear();

    if (rank == 0)
        std::cout << "Distributing precomputation data throughout ranks\n";

    // Make a new matrix with rows distributed among MPI ranks evenly
    RCP<const map_type> global_map = rcp(new map_type(nrows, indexBase, comm, Tpetra::GloballyDistributed));
    Tpetra::Export<> exporter(rank_0_map, global_map);
    // Redistribute the data, NOT in place, from matrices on rank 0
    // to distributed versions (which are distributed evenly over the processes).
    M_inv_ = rcp(new matrix_type(global_map, ncols));
    M_inv_->doExport(*rank_0_M_inv, exporter, Tpetra::INSERT);
    stresslet_ = rcp(new matrix_type(global_map, ncols));
    stresslet_->doExport(*rank_0_stresslet, exporter, Tpetra::INSERT);

    if (rank == 0)
        std::cout << "Done initializing periphery\n";
}
