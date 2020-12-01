#include <cnpy.hpp>
#include <kernels.hpp>
#include <periphery.hpp>

#include <mpi.h>

Eigen::MatrixXd Periphery::flow(const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                                const Eigen::Ref<const Eigen::MatrixXd> &density, double eta) const {
    // FIXME: Move fmm object and make more flexible
    static kernels::FMM<stkfmm::Stk3DFMM> fmm(8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel,
                                              kernels::stokes_pvel_fmm);

    const int n_dl = density.size() / 3;
    const int n_trg = r_trg.size() / 3;
    Eigen::MatrixXd f_dl(9, n_dl);

    Eigen::Map<const Eigen::MatrixXd> density_reshaped(density.data(), 3, n_dl);

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_dl; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normal_(i, node) * density_reshaped(j, node);

    Eigen::MatrixXd r_sl, f_sl; // dummy SL positions/values
    // FIXME: Why is this line necessary?
    Eigen::MatrixXd r_dl = node_pos_; // Double layer coordinates are node positions
    Eigen::MatrixXd pvel = fmm(r_sl, r_dl, r_trg, f_sl, f_dl);
    Eigen::MatrixXd vel = pvel.block(1, 0, 3, n_trg) / eta;

    return vel;
}

Periphery::Periphery(const std::string &precompute_file) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    cnpy::npz_t precomp;

    if (world_rank == 0)
        std::cout << "Loading raw precomputation data from file " << precompute_file << " for periphery into rank 0\n";
    int n_rows;
    int n_nodes;
    if (world_rank == 0) {
        precomp = cnpy::npz_load(precompute_file);
        n_rows = precomp.at("M_inv").shape[0];
        n_nodes = precomp.at("nodes").shape[0];
    }

    MPI_Bcast((void *)&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *)&n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_cols = n_rows;
    const int node_size_big = 3 * (n_nodes / world_size + 1);
    const int node_size_small = 3 * (n_nodes / world_size);
    const int node_size_local = (n_nodes % world_size > world_rank) ? node_size_big : node_size_small;
    const int n_nodes_big = n_nodes % world_size;
    const int nrows_local = node_size_local;

    // TODO: prevent overflow for large matrices in periphery import
    node_counts_.resize(world_size);
    node_displs_ = Eigen::VectorXi::Zero(world_size + 1);
    for (int i = 0; i < world_size; ++i) {
        node_counts_[i] = ((i < n_nodes_big) ? node_size_big : node_size_small);
        node_displs_[i + 1] = node_displs_[i] + node_counts_[i];
    }
    row_counts_ = n_cols * node_counts_;
    row_displs_ = n_cols * node_displs_;
    quad_counts_ = node_counts_ / 3;
    quad_displs_ = node_displs_ / 3;

    const double *M_inv_raw = (world_rank == 0) ? precomp["M_inv"].data<double>() : NULL;
    const double *stresslet_plus_complementary_raw =
        (world_rank == 0) ? precomp["stresslet_plus_complementary"].data<double>() : NULL;
    const double *normals_raw = (world_rank == 0) ? precomp["normals"].data<double>() : NULL;
    const double *nodes_raw = (world_rank == 0) ? precomp["nodes"].data<double>() : NULL;
    const double *quadrature_weights_raw = (world_rank == 0) ? precomp["quadrature_weights"].data<double>() : NULL;

    // Numpy data is row-major, while eigen is column-major. Easiest way to rectify this is to
    // load in matrix as its transpose, then transpose back
    M_inv_.resize(n_cols, nrows_local);
    MPI_Scatterv(M_inv_raw, row_counts_.data(), row_displs_.data(), MPI_DOUBLE, M_inv_.data(), row_counts_[world_rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    stresslet_plus_complementary_.resize(n_cols, nrows_local);
    MPI_Scatterv(M_inv_raw, row_counts_.data(), row_displs_.data(), MPI_DOUBLE, stresslet_plus_complementary_.data(),
                 row_counts_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    M_inv_ = M_inv_.transpose();
    stresslet_plus_complementary_ = stresslet_plus_complementary_.transpose();

    node_normal_.resize(3, node_size_local / 3);
    MPI_Scatterv(normals_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_normal_.data(),
                 node_counts_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    node_pos_.resize(3, node_size_local / 3);
    MPI_Scatterv(nodes_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_pos_.data(),
                 node_counts_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    quadrature_weights_.resize(node_size_local / 3);
    MPI_Scatterv(quadrature_weights_raw, quad_counts_.data(), quad_displs_.data(), MPI_DOUBLE,
                 quadrature_weights_.data(), quad_counts_[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    n_nodes_global_ = n_nodes;

    if (world_rank == 0)
        std::cout << "Done initializing periphery\n";
}
