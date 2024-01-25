#include <body.hpp>
#include <body_deformable.hpp>
#include <body_ellipsoidal.hpp>
#include <body_spherical.hpp>
#include <cnpy.hpp>
#include <fiber_container_finite_difference.hpp>
#include <fiber_finite_difference.hpp>
#include <kernels.hpp>
#include <periphery.hpp>
#include <system.hpp>
#include <utils.hpp>

#include <spdlog/fmt/ostr.h>

/// @brief Apply preconditioner for Periphery component of 'x'.  While local input is supplied,
/// the preconditioner result requires the 'global' set of 'x' across all ranks, so an
/// Allgatherv is required
///
/// @param[in] x_local [3 * n_nodes_local] vector of 'x' local to this rank
/// @return [3 * n_nodes_local] vector of P * x_local
Eigen::VectorXd Periphery::apply_preconditioner(CVectorRef &x_local) const {
    if (!n_nodes_global_)
        return Eigen::VectorXd();
    assert(x_local.size() == get_local_solution_size());
    Eigen::VectorXd x_shell(3 * n_nodes_global_);
    MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                   node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return M_inv_ * x_shell;
}

/// @brief Apply matvec for Periphery component of 'x'.  While local input is supplied,
/// the matvec result requires the 'global' set of 'x' across all ranks, so an
/// Allgatherv is required (though not on v)
///
/// @param[in] x_local [3 * n_nodes_local] vector of 'x' local to this rank
/// @param[in] v_local [3 * n_nodes_local] vector of velocities 'v' local to this rank
/// @return [3 * n_nodes_local] vector of A * x_local
Eigen::VectorXd Periphery::matvec(CVectorRef &x_local, CMatrixRef &v_local) const {
    if (!n_nodes_global_)
        return Eigen::VectorXd();
    assert(x_local.size() == get_local_solution_size());
    assert(v_local.size() == get_local_solution_size());
    Eigen::VectorXd x_shell(3 * n_nodes_global_);
    MPI_Allgatherv(x_local.data(), node_counts_[world_rank_], MPI_DOUBLE, x_shell.data(), node_counts_.data(),
                   node_displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    return stresslet_plus_complementary_ * x_shell + CVectorMap(v_local.data(), v_local.size());
}

/// @brief Calculate velocity at target coordinates due to the periphery
/// Input:
/// @param[in] r_trg [3 x n_trg_local] matrix of target coordinates to evaluate the velocity at
/// @param[in] density [3 x n_nodes_local] matrix of node source strengths
/// @param[in] eta fluid viscosity
/// @return [3 x n_trg_local] matrix of velocity at target coordinates
Eigen::MatrixXd Periphery::flow(CMatrixRef &r_trg, CMatrixRef &density, double eta) const {
    spdlog::debug("Started shell flow");
    if (!n_nodes_global_)
        return Eigen::MatrixXd::Zero(3, r_trg.cols());
    utils::LoggerRedirect redirect(std::cout);
    const int n_dl = density.size() / 3;
    Eigen::MatrixXd f_dl(9, n_dl);

    CMatrixMap density_reshaped(density.data(), 3, n_dl);

    // double layer density is 2 * outer product of normals with density
    // scales with viscosity since the stresslet_kernel_ routine divides by the viscosity, and the double-layer
    // stresslet is independent of viscosity
    for (int node = 0; node < n_dl; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * eta * node_normal_(i, node) * density_reshaped(j, node);

    Eigen::MatrixXd r_sl, f_sl; // dummy SL positions/values
    Eigen::MatrixXd vel = stresslet_kernel_(r_sl, node_pos_, r_trg, f_sl, f_dl, eta);
    redirect.flush(spdlog::level::debug, "STKFMM");

    spdlog::debug("Finished shell flow");
    return vel;
}

/// @brief Update the internal right-hand-side state given the velocity at the shell nodes
/// No prerequisite calculations, beyond initialization, are needed
///
/// @param[in] v_on_shell [3 x n_nodes_local] matrix of velocity at shell nodes on local to this MPI rank
/// @return true if collision, false otherwise
void Periphery::update_RHS(CMatrixRef &v_on_shell) { RHS_ = -CVectorMap(v_on_shell.data(), v_on_shell.size()); }

/// @brief Check for collision between SphericalPeriphery and SphericalBody
/// If any point on body > (this->radius_ - threshold), then a collision is detected
///
/// @param[in] body SphericalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return true if collision, false otherwise
bool SphericalPeriphery::check_collision(const SphericalBody &body, double threshold) const {
    const double max_distance = body.position_.norm() + body.radius_;
    return max_distance > (radius_ - threshold);
}

/// @brief STUB Check for collision between SphericalPeriphery and DeformableBody
///
/// @param[in] body DeformableBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false because it's not implemented
bool SphericalPeriphery::check_collision(const DeformableBody &body, double threshold) const {
    spdlog::warn("check_collision not implemented for SphericalPeriphery->DeformableBody");
    return false;
}

/// @brief STUB Check for collision between SphericalPeriphery and EllipsoidalBody
///
/// @param[in] body EllipsoidalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false because it's not implemented
bool SphericalPeriphery::check_collision(const EllipsoidalBody &body, double threshold) const {
    spdlog::warn("check_collision not implemented for SphericalPeriphery->EllipsoidalBody");
    return false;
}

/// @brief Check for collision between SphericalPeriphery and a point cloud
/// If any point lies outside R=(this->radius_ - threshold), return true
/// Useful for collision detection between fibers and the periphery, but could be primitively used for a DeformableBody
///
/// @param[in] point_cloud [3 x n_points] matrix of points to check collision
/// @param[in] threshold signed threshold to check collision
/// @return true if collision, false otherwise
bool SphericalPeriphery::check_collision(const CMatrixRef &point_cloud, double threshold) const {
    const double r2 = pow(radius_ - threshold, 2);
    for (int i = 0; i < point_cloud.cols(); ++i)
        if (point_cloud.col(i).squaredNorm() >= r2)
            return true;

    return false;
}

/// @brief Calculate steric forces between SphericalPeriphery and a fiber
///
/// @param[in] fiber to interact with periphery
/// @param[in] fp_params structure which parameterizes this interaction
/// @return [3 x n_points] matrix of forces on points due to the Periphery
Eigen::MatrixXd SphericalPeriphery::fiber_interaction(const FiberFiniteDifference &fiber,
                                                      const fiber_periphery_interaction_t &fp_params) const {
    if (!n_nodes_global_)
        return Eigen::MatrixXd::Zero(fiber.x_.rows(), fiber.x_.cols());

    const CMatrixRef &pc = fiber.x_;
    Eigen::MatrixXd f_points = Eigen::MatrixXd::Zero(pc.rows(), pc.cols());

    const int start_index = fiber.minus_clamped_ ? 1 : 0;
    for (int i = start_index; i < pc.cols(); ++i) {
        double r_mag = pc.col(i).norm();

        if (r_mag < radius_) {
            Eigen::VectorXd u_hat = pc.col(i) / r_mag;
            Eigen::Vector3d dr = pc.col(i) - u_hat * radius_;
            double d = dr.norm();
            f_points.col(i) = fp_params.f_0 * dr / d * exp(-(radius_ - r_mag) / fp_params.l_0);
        } else
            spdlog::debug("FiberFiniteDifference collision detected in force routine.");
    }

    return f_points;
}

/// @brief STUB Check for collision between EllipsoidalPeriphery and SphericalBody
///
/// @param[in] body SphericalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool EllipsoidalPeriphery::check_collision(const SphericalBody &body, double threshold) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("check_collision not implemented for EllipsoidalPeriphery->SphericalBody");
        first_call = false;
    }
    return false;
}

/// @brief STUB Check for collision between EllipsoidalPeriphery and DeformableBody
///
/// @param[in] body DeformableBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool EllipsoidalPeriphery::check_collision(const DeformableBody &body, double threshold) const {
    spdlog::warn("check_collision not implemented for EllipsoidalPeriphery->DeformableBody");
    return false;
}

/// @brief STUB Check for collision between EllipsoidalPeriphery and EllipsoidalBody
///
/// @param[in] body EllipsoidalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool EllipsoidalPeriphery::check_collision(const EllipsoidalBody &body, double threshold) const {
    spdlog::warn("check_collision not implemented for EllipsoidalPeriphery->EllipsoidalBody");
    return false;
}

/// @brief Check for collision between EllipsoidalPeriphery and a point cloud
/// Useful for collision detection between fibers and the periphery, but could be primitively used for a DeformableBody
///
/// @param[in] point_cloud [3 x n_points] matrix of points to check collision
/// @param[in] threshold signed threshold to check collision
/// @return true if collision, false otherwise
bool EllipsoidalPeriphery::check_collision(const CMatrixRef &point_cloud, double threshold) const {
    const CMatrixRef &pc = point_cloud;
    for (int i = 0; i < pc.cols(); ++i) {
        Eigen::Vector3d r_scaled = pc.col(i).array() / Eigen::Array3d{a_, b_, c_};
        double r_scaled_mag = r_scaled.norm();
        double phi = atan2(r_scaled.y(), (r_scaled.x() + 1E-12));
        double theta = acos(r_scaled.z() / (1E-12 + r_scaled_mag));
        double sintheta = sin(theta);

        Eigen::Vector3d r_cortex{(a_ - threshold) * sintheta * cos(phi), (b_ - threshold) * sintheta * sin(phi),
                                 (c_ - threshold) * cos(theta)};
        if (pc.col(i).squaredNorm() >= r_cortex.squaredNorm()) {
            spdlog::debug("EllipsoidalPeriphery and FiberFiniteDifference collision, fiber point [{}] (may be adjusted "
                          "due to clamping)",
                          i);
            return true;
        }
    }

    return false;
}

/// @brief Calculate steric forces between SphericalPeriphery and a fiber
///
/// @param[in] fiber to interact with periphery
/// @param[in] point_cloud [3 x n_points] matrix of points to interact with periphery
/// @param[in] fp_params structure which parameterizes this interaction
/// @return [3 x n_points] matrix of forces on points due to the Periphery
Eigen::MatrixXd EllipsoidalPeriphery::fiber_interaction(const FiberFiniteDifference &fiber,
                                                        const fiber_periphery_interaction_t &fp_params) const {
    const CMatrixRef &pc = fiber.x_;
    if (!n_nodes_global_)
        return Eigen::MatrixXd::Zero(pc.rows(), pc.cols());

    Eigen::MatrixXd f_points = Eigen::MatrixXd::Zero(pc.rows(), pc.cols());

    const int start_index = fiber.minus_clamped_ ? 1 : 0;
    for (int i = start_index; i < pc.cols(); ++i) {
        Eigen::Vector3d r_scaled = pc.col(i).array() / Eigen::Array3d{a_, b_, c_};
        double r_scaled_mag = r_scaled.norm();
        double r_mag = pc.col(i).norm();

        double phi = atan2(r_scaled.y(), (r_scaled.x() + 1E-12));
        double theta = acos(r_scaled.z() / (1E-12 + r_scaled_mag));
        double sintheta = sin(theta);

        Eigen::Vector3d r_cortex{a_ * sintheta * cos(phi), b_ * sintheta * sin(phi), c_ * cos(theta)};

        double r_cortex_mag = r_cortex.norm();
        if (r_mag < r_cortex_mag) {
            Eigen::Vector3d dr = pc.col(i) - r_cortex;
            double d = dr.norm();
            f_points.col(i) = fp_params.f_0 * dr / d * exp(-(r_cortex_mag - r_mag) / fp_params.l_0);
        } else {
            spdlog::debug("FiberFiniteDifference collision detected in force routine.");
        }
    }

    return f_points;
}

/// @brief STUB Check for collision between GenericPeriphery and SphericalBody
///
/// @param[in] body SphericalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool GenericPeriphery::check_collision(const SphericalBody &body, double threshold) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("check_collision not implemented for GenericPeriphery->SphericalBody");
        first_call = false;
    }
    return false;
}

/// @brief STUB Check for collision between GenericPeriphery and SphericalBody
///
/// @param[in] body DeformableBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool GenericPeriphery::check_collision(const DeformableBody &body, double threshold) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("check_collision not implemented for GenericPeriphery->DeformableBody");
        first_call = false;
    }
    return false;
}

/// @brief STUB Check for collision between GenericPeriphery and EllipsoidalBody
///
/// @param[in] body EllipsoidalBody to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool GenericPeriphery::check_collision(const EllipsoidalBody &body, double threshold) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("check_collision not implemented for GenericPeriphery->EllipsoidalBody");
        first_call = false;
    }
    return false;
}

/// @brief STUB Check for collision between GenericPeriphery and point_cloud
///
/// @param[in] body CMatrixRef (point cloud) to check collision
/// @param[in] threshold signed threshold to check collision
/// @return always false
bool GenericPeriphery::check_collision(const CMatrixRef &point_cloud, double threshold) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("check_collision not implemented for point clouds");
        first_call = false;
    }
    return false;
}

/// @brief STUB Calculate forces between GenericPeriphery and FiberFiniteDifference
///
/// @param[in] fiber to interact with periphery
/// @param[in] fp_params structure which parameterizes this interaction
/// @return [3 x n_points] matrix of forces (ZEROS) on points due to the Periphery
Eigen::MatrixXd GenericPeriphery::fiber_interaction(const FiberFiniteDifference &fiber,
                                                    const fiber_periphery_interaction_t &fp_params) const {
    static bool first_call = true;
    if (!world_rank_ && first_call) {
        spdlog::warn("fiber_interaction_finitediff not implemented for GenericPeriphery->FiberFiniteDifference");
        first_call = false;
    }

    return Eigen::MatrixXd::Zero(fiber.x_.rows(), fiber.x_.cols());
}

void Periphery::set_evaluator(const std::string &evaluator) {
    auto &params = *System::get_params();

    if (evaluator == "FMM") {
        using namespace kernels;
        using namespace stkfmm;
        const int mult_order = params.stkfmm.periphery_stresslet_multipole_order;
        const int max_pts = params.stkfmm.periphery_stresslet_max_points;
        utils::LoggerRedirect redirect(std::cout);
        stresslet_kernel_ = FMM<Stk3DFMM>(mult_order, max_pts, PAXIS::NONE, KERNEL::PVel, stokes_pvel_fmm);
        redirect.flush(spdlog::level::debug, "STKFMM");
    } else if (evaluator == "CPU")
        stresslet_kernel_ = kernels::stresslet_direct_cpu;
    else if (evaluator == "GPU")
        stresslet_kernel_ = kernels::stresslet_direct_gpu;
}

/// @brief Construct Periphery base class object
///
/// @param[in] precompute_file '.npz' file generated by precompute script
/// @param[in] periphery_table parsed toml object representing periphery config
/// @param[in] params system Params struct
Periphery::Periphery(const toml::value &periphery_table, const Params &params) {

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

    set_evaluator(params.pair_evaluator);

    cnpy::npz_t precomp;
    const std::string precompute_file = toml::find_or(periphery_table, "precompute_file", "");
    if (!precompute_file.length())
        throw std::runtime_error(
            "Periphery specified, but no precompute file. In your config file under [periphery], "
            "set precompute_file and run 'skelly_precompute' on the config."
            "If using the config generator, it should automatically generate this variable, though "
            "you still need to run the precompute script after generating the config.");

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

    solution_vec_ = Eigen::VectorXd::Zero(nrows_local);

    // TODO: prevent overflow for large matrices in periphery import
    node_counts_.resize(world_size_);
    node_displs_ = Eigen::VectorXi::Zero(world_size_ + 1);
    for (int i = 0; i < world_size_; ++i) {
        node_counts_[i] = ((i < n_nodes_big) ? node_size_big : node_size_small);
        node_displs_[i + 1] = node_displs_[i] + node_counts_[i];
    }
    row_counts_ = node_counts_;
    row_displs_ = node_displs_;
    quad_counts_ = node_counts_ / 3;
    quad_displs_ = node_displs_ / 3;

    const double *M_inv_raw = (world_rank_ == 0) ? precomp["M_inv"].data<double>() : NULL;
    const double *stresslet_plus_complementary_raw =
        (world_rank_ == 0) ? precomp["stresslet_plus_complementary"].data<double>() : NULL;
    const double *normals_raw = (world_rank_ == 0) ? precomp["normals"].data<double>() : NULL;
    const double *nodes_raw = (world_rank_ == 0) ? precomp["nodes"].data<double>() : NULL;
    const double *quadrature_weights_raw = (world_rank_ == 0) ? precomp["quadrature_weights"].data<double>() : NULL;

    MPI_Datatype mpi_matrix_row_t;
    MPI_Type_contiguous(n_cols, MPI_DOUBLE, &mpi_matrix_row_t);
    MPI_Type_commit(&mpi_matrix_row_t);

    // Numpy data is row-major, while eigen is column-major. Easiest way to rectify this is to
    // load in matrix as its transpose, then transpose back
    M_inv_.resize(n_cols, nrows_local);
    MPI_Scatterv(M_inv_raw, row_counts_.data(), row_displs_.data(), mpi_matrix_row_t, M_inv_.data(),
                 row_counts_[world_rank_], mpi_matrix_row_t, 0, MPI_COMM_WORLD);

    stresslet_plus_complementary_.resize(n_cols, nrows_local);
    MPI_Scatterv(stresslet_plus_complementary_raw, row_counts_.data(), row_displs_.data(), mpi_matrix_row_t,
                 stresslet_plus_complementary_.data(), row_counts_[world_rank_], mpi_matrix_row_t, 0, MPI_COMM_WORLD);

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

    MPI_Type_free(&mpi_matrix_row_t);

    // Print functionality
    spdlog::info("Periphery constructed");
    spdlog::info("  Periphery type: {}", toml::find(periphery_table, "shape"));

    spdlog::info("Done initializing base periphery");
}
