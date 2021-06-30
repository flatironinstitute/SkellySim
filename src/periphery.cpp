#include <body.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <periphery.hpp>
#include <utils.hpp>

#include <mpi.h>

#include <spdlog/spdlog.h>

Eigen::VectorXd Periphery::apply_preconditioner(VectorRef &x_local) const {
    if (!n_nodes_global_)
        return Eigen::VectorXd();
    assert(x_local.size() == get_local_solution_size());
    Eigen::VectorXd x_shell(3 * n_nodes_global_);
    MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                   node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return M_inv_ * x_shell;
}

Eigen::VectorXd Periphery::matvec(VectorRef &x_local, MatrixRef &v_local) const {
    if (!n_nodes_global_)
        return Eigen::VectorXd();
    assert(x_local.size() == get_local_solution_size());
    assert(v_local.size() == get_local_solution_size());
    Eigen::VectorXd x_shell(3 * n_nodes_global_);
    MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                   node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return stresslet_plus_complementary_ * x_shell + CVectorMap(v_local.data(), v_local.size());
}

Eigen::MatrixXd Periphery::flow(MatrixRef &r_trg, MatrixRef &density, double eta) const {
    // Calculate velocity at target coordinates due to the periphery.
    // Input:
    //    const r_trg [3xn_trg_local]: Target coordinates
    //    const density [3*n_nodes_local]: Strength of node sources
    //    eta: Fluid viscosity
    // Output:
    //    vel [3xn_trg_local]: velocity at target coordinates
    spdlog::debug("Started shell flow");
    if (!n_nodes_global_)
        return Eigen::MatrixXd::Zero(3, r_trg.cols());
    utils::LoggerRedirect redirect(std::cout);
    const int n_dl = density.size() / 3;
    const int n_trg = r_trg.size() / 3;
    Eigen::MatrixXd f_dl(9, n_dl);

    CMatrixMap density_reshaped(density.data(), 3, n_dl);

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_dl; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normal_(i, node) * density_reshaped(j, node);

    Eigen::MatrixXd r_sl, f_sl; // dummy SL positions/values
    Eigen::MatrixXd pvel = (*stresslet_kernel_)(r_sl, node_pos_, r_trg, f_sl, f_dl);
    Eigen::MatrixXd vel = pvel.block(1, 0, 3, n_trg) / eta;
    redirect.flush(spdlog::level::debug, "STKFMM");

    spdlog::debug("Finished shell flow");
    return vel;
}

void Periphery::update_RHS(MatrixRef &v_on_shell) {
    // Update the internal right-hand-side state.
    // No prerequisite calculations, beyond initialization, are needed
    // Input:
    //    const v_on_shell [3xn_nodes_local]: Velocity at shell nodes on local MPI rank
    RHS_ = -CVectorMap(v_on_shell.data(), v_on_shell.size());
}

bool SphericalPeriphery::check_collision(const SphericalBody &body, double threshold) const {
    const double max_distance = body.position_.norm() + body.radius_;
    return max_distance > (radius_ - threshold);
}

bool SphericalPeriphery::check_collision(const DeformableBody &body, double threshold) const {
    spdlog::warn("check_collision not implemented for SphericalPeriphy->DeformableBody");
    return false;
}

bool SphericalPeriphery::check_collision(const MatrixRef &point_cloud, double threshold) const {
    const double r2 = pow(radius_ - threshold, 2);
    for (int i = 0; i < point_cloud.cols(); ++i)
        if (point_cloud.col(i).squaredNorm() >= r2)
            return true;

    return false;
}

Eigen::MatrixXd SphericalPeriphery::point_cloud_interaction(const MatrixRef &point_cloud,
                                                            const fiber_periphery_interaction_t &fp_params) const {
    if (!n_nodes_global_)
        return Eigen::MatrixXd::Zero(point_cloud.rows(), point_cloud.cols());

    const MatrixRef &pc = point_cloud;
    Eigen::MatrixXd f_points = Eigen::MatrixXd::Zero(pc.rows(), pc.cols());

    for (int i = 0; i < pc.cols(); ++i) {
        double r_mag = pc.col(i).norm();

        Eigen::VectorXd u_hat = pc.col(i) / r_mag;
        Eigen::Vector3d dr = pc.col(i) - u_hat * radius_;
        double d = dr.norm();

        if (r_mag < radius_)
            f_points.col(i) = fp_params.f_0 * dr / d * exp(-(radius_ - r_mag) / fp_params.lambda);
        else
            spdlog::debug("Fiber collision detected in force routine.");
    }

    return f_points;
}

Periphery::Periphery(const std::string &precompute_file, const toml::value &body_table, const Params &params) {
    {
        using namespace kernels;
        using namespace stkfmm;
        const int mult_order = params.stkfmm.periphery_stresslet_multipole_order;
        const int max_pts = params.stkfmm.periphery_stresslet_max_points;
        utils::LoggerRedirect redirect(std::cout);
        stresslet_kernel_ = std::unique_ptr<FMM<Stk3DFMM>>(
            new FMM<Stk3DFMM>(mult_order, max_pts, PAXIS::NONE, KERNEL::PVel, stokes_pvel_fmm));
        redirect.flush(spdlog::level::debug, "STKFMM");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

    cnpy::npz_t precomp;

    spdlog::info("Loading raw precomputation data from file {} for periphery into rank 0", precompute_file);
    int n_rows;
    int n_nodes;
    if (world_rank_ == 0) {
        precomp = cnpy::npz_load(precompute_file);
        n_rows = precomp.at("M_inv").shape[0];
        n_nodes = precomp.at("nodes").shape[0];
    }

    MPI_Bcast((void *)&n_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast((void *)&n_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int n_cols = n_rows;
    const int node_size_big = 3 * (n_nodes / world_size_ + 1);
    const int node_size_small = 3 * (n_nodes / world_size_);
    const int node_size_local = (n_nodes % world_size_ > world_rank_) ? node_size_big : node_size_small;
    const int n_nodes_big = n_nodes % world_size_;
    const int nrows_local = node_size_local;

    // TODO: prevent overflow for large matrices in periphery import
    node_counts_.resize(world_size_);
    node_displs_ = Eigen::VectorXi::Zero(world_size_ + 1);
    for (int i = 0; i < world_size_; ++i) {
        node_counts_[i] = ((i < n_nodes_big) ? node_size_big : node_size_small);
        node_displs_[i + 1] = node_displs_[i] + node_counts_[i];
    }
    row_counts_ = n_cols * node_counts_;
    row_displs_ = n_cols * node_displs_;
    quad_counts_ = node_counts_ / 3;
    quad_displs_ = node_displs_ / 3;

    const double *M_inv_raw = (world_rank_ == 0) ? precomp["M_inv"].data<double>() : NULL;
    const double *stresslet_plus_complementary_raw =
        (world_rank_ == 0) ? precomp["stresslet_plus_complementary"].data<double>() : NULL;
    const double *normals_raw = (world_rank_ == 0) ? precomp["normals"].data<double>() : NULL;
    const double *nodes_raw = (world_rank_ == 0) ? precomp["nodes"].data<double>() : NULL;
    const double *quadrature_weights_raw = (world_rank_ == 0) ? precomp["quadrature_weights"].data<double>() : NULL;

    // Numpy data is row-major, while eigen is column-major. Easiest way to rectify this is to
    // load in matrix as its transpose, then transpose back
    M_inv_.resize(n_cols, nrows_local);
    MPI_Scatterv(M_inv_raw, row_counts_.data(), row_displs_.data(), MPI_DOUBLE, M_inv_.data(), row_counts_[world_rank_],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    stresslet_plus_complementary_.resize(n_cols, nrows_local);
    MPI_Scatterv(stresslet_plus_complementary_raw, row_counts_.data(), row_displs_.data(), MPI_DOUBLE,
                 stresslet_plus_complementary_.data(), row_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    M_inv_.transposeInPlace();
    stresslet_plus_complementary_.transposeInPlace();

    node_normal_.resize(3, node_size_local / 3);
    MPI_Scatterv(normals_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_normal_.data(),
                 node_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    node_pos_.resize(3, node_size_local / 3);
    MPI_Scatterv(nodes_raw, node_counts_.data(), node_displs_.data(), MPI_DOUBLE, node_pos_.data(),
                 node_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    quadrature_weights_.resize(node_size_local / 3);
    MPI_Scatterv(quadrature_weights_raw, quad_counts_.data(), quad_displs_.data(), MPI_DOUBLE,
                 quadrature_weights_.data(), quad_counts_[world_rank_], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    n_nodes_global_ = n_nodes;

    spdlog::info("Done initializing base periphery");
}
