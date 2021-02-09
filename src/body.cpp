#include <body.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <parse_util.hpp>

/// @brief Update internal Body::K_ matrix variable
/// @see update_preconditioner
void Body::update_K_matrix() {
    K_.resize(3 * n_nodes_, 6);
    for (int i = 0; i < n_nodes_; ++i) {
        // J matrix
        K_.block(i * 3, 0, 3, 3).diagonal().array() = -1.0;
        // rot matrix
        Eigen::Vector3d vec = node_positions_.col(i);
        K_.block(i * 3 + 0, 3, 1, 3) = -Eigen::RowVector3d({0.0, vec[2], -vec[1]});
        K_.block(i * 3 + 1, 3, 1, 3) = -Eigen::RowVector3d({-vec[2], 0.0, vec[0]});
        K_.block(i * 3 + 2, 3, 1, 3) = -Eigen::RowVector3d({vec[1], -vec[0], 0.0});
    }
}

/// @brief Update internal variables that need to be recomputed only once after calling Body::move
///
/// @see update_singularity_subtraction_vecs
/// @see update_K_matrix
/// @see update_preconditioner
/// @param[in] eta fluid viscosity
void Body::update_cache_variables(double eta) {
    update_singularity_subtraction_vecs(eta);
    update_K_matrix();
    update_preconditioner(eta);
}

/// @brief Update the preconditioner and associated linear operator
///
/// Updates: Body::A_, Body::_LU_
/// @param[in] eta
void Body::update_preconditioner(double eta) {
    A_.resize(3 * n_nodes_ + 6, 3 * n_nodes_ + 6);
    A_.setZero();

    // M matrix
    A_.block(0, 0, 3 * n_nodes_, 3 * n_nodes_) = kernels::stresslet_times_normal(node_positions_, node_normals_, eta);

    for (int i = 0; i < n_nodes_; ++i) {
        A_.block(i * 3, 3 * i + 0, 3, 1) -= ex_.col(i) / node_weights_[i];
        A_.block(i * 3, 3 * i + 1, 3, 1) -= ey_.col(i) / node_weights_[i];
        A_.block(i * 3, 3 * i + 2, 3, 1) -= ez_.col(i) / node_weights_[i];
    }

    // K matrix
    A_.block(0, 3 * n_nodes_, 3 * n_nodes_, 6) = -K_;

    // K^T matrix
    A_.block(3 * n_nodes_, 0, 6, 3 * n_nodes_) = -K_.transpose();

    // Last block is apparently diagonal.
    A_.block(3 * n_nodes_, 3 * n_nodes_, 6, 6).diagonal().array() = 1.0;

    A_LU_.compute(A_);
}

/// @brief Calculate the current RHS_, given the current velocity on the body's nodes
///
/// Updates only Body::RHS_
/// \f[ \textrm{RHS} = -\left[{\bf v}_0, {\bf v}_1, ..., {\bf v}_N \right] \f]
/// @param[in] v_on_body [ 3 x n_nodes ] matrix representing the velocity on each node
void Body::update_RHS(const Eigen::Ref<const Eigen::MatrixXd> v_on_body) {
    RHS_ = -Eigen::Map<const Eigen::VectorXd>(v_on_body.data(), v_on_body.size());
}

/// @brief Move body to new position with new orientation
///
/// Updates: Body::position_, Body::orientation_, Body::node_positions_, Body::node_normals_, Body::nucleation_sites_
/// @param[in] new_pos new lab frame position to move the body centroid
/// @param[in] new_orientation new orientation of the body
void Body::move(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation) {
    position_ = new_pos;
    orientation_ = new_orientation;

    Eigen::Matrix3d rot = orientation_.toRotationMatrix();
    for (int i = 0; i < n_nodes_; ++i)
        node_positions_.col(i) = position_ + rot * node_positions_ref_.col(i);

    for (int i = 0; i < n_nodes_; ++i)
        node_normals_.col(i) = rot * node_normals_ref_.col(i);

    for (int i = 0; i < n_nodes_; ++i)
        nucleation_sites_ref_.col(i) = position_ + rot * nucleation_sites_ref_.col(i);
}

/// @brief Calculate/cache the internal 'singularity subtraction vectors', used in linear operator application
///
/// Need to call after calling Body::move.
/// @see update_preconditioner
///
/// Updates: Body::ex_, Body::ey_, Body::ez_
/// @param[in] eta viscosity of fluid
void Body::update_singularity_subtraction_vecs(double eta) {
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(3, n_nodes_);

    e.row(0) = node_weights_.transpose();
    ex_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);

    e.row(0).array() = 0.0;
    e.row(1) = node_weights_.transpose();
    ey_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);

    e.row(1).array() = 0.0;
    e.row(2) = node_weights_.transpose();
    ez_ = kernels::stresslet_times_normal_times_density(node_positions_, node_normals_, e, eta);
}

/// @brief Helper function that loads precompute data.
///
/// Data is loaded on _every_ MPI rank, and should be identical.
/// Updates: Body::node_positions_ref_, Body::node_normals_ref_, Body::node_weights_
///   @param[in] precompute_file path to file containing precompute data in npz format (from numpy.save). See associated
///   utility precompute script `utils/make_precompute_data.py`
void Body::load_precompute_data(const std::string &precompute_file) {
    cnpy::npz_t precomp = cnpy::npz_load(precompute_file);
    auto load_mat = [](cnpy::npz_t &npz, const char *var) {
        return Eigen::Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0]).matrix();
    };

    auto load_vec = [](cnpy::npz_t &npz, const char *var) {
        return Eigen::Map<Eigen::VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
    };

    node_positions_ref_ = load_mat(precomp, "node_positions_ref");
    node_normals_ref_ = load_mat(precomp, "node_normals_ref");
    node_weights_ = load_vec(precomp, "node_weights");
}

/// @brief Construct body from relevant toml config and system params
///
///   @param[in] body_table toml table from pre-parsed config
///   @param[in] params Pre-constructed Params object
///   surface).
///   @return Body object that has been appropriately rotated. Other internal cache variables are _not_ updated.
/// @see update_cache_variables
Body::Body(const toml::table *body_table, const Params &params) {
    using namespace parse_util;
    using std::string;
    string precompute_file = parse_val_key<string>(body_table, "precompute_file");
    load_precompute_data(precompute_file);

    // TODO: add body assertions so that input file and precompute data necessarily agree
    n_nodes_ = node_positions_.cols();

    if (!!body_table->get("position"))
        position_ = parse_array_key<>(body_table, "position");

    if (!!body_table->get("orientation"))
        orientation_ = parse_array_key<Eigen::Quaterniond>(body_table, "orientation");

    if (!!body_table->get("nucleation_sites")) {
        nucleation_sites_ref_ = parse_array_key<>(body_table, "nucleation_sites");
        nucleation_sites_ref_.resize(3, nucleation_sites_ref_.size() / 3);
    }

    move(position_, orientation_);
}

Eigen::VectorXd BodyContainer::get_RHS() const {
    Eigen::VectorXd RHS(get_local_solution_size());

    if (world_rank_ == 0) {
        int offset = 0;
        for (const auto &body : bodies) {
            RHS.segment(offset, body.RHS_.size()) = body.RHS_;
            offset += body.RHS_.size();
        }
    }
    return RHS;
}

Eigen::MatrixXd BodyContainer::get_node_positions() const {
    Eigen::MatrixXd r_body_nodes;

    const int n_nodes = get_local_solution_size();
    if (world_rank_ == 0) {
        r_body_nodes.resize(3, n_nodes);
        int offset = 0;
        for (const auto &body : bodies) {
            r_body_nodes.block(0, offset, 3, body.n_nodes_) = body.node_positions_;
            offset += body.n_nodes_;
        }
    }

    return r_body_nodes;
}

Eigen::MatrixXd BodyContainer::get_node_normals() const {
    Eigen::MatrixXd r_body_nodes;

    const int n_nodes = get_local_solution_size();
    if (world_rank_ == 0) {
        r_body_nodes.resize(3, n_nodes);
        int offset = 0;
        for (const auto &body : bodies) {
            r_body_nodes.block(0, offset, 3, body.n_nodes_) = body.node_normals_;
            offset += body.n_nodes_;
        }
    }

    return r_body_nodes;
}

Eigen::MatrixXd BodyContainer::flow(const Eigen::Ref<const Eigen::MatrixXd> &r_trg,
                                    const Eigen::Ref<const Eigen::MatrixXd> &densities,
                                    const Eigen::Ref<const Eigen::MatrixXd> &forces_torques, double eta) const {
    const int n_nodes = get_local_node_count();               //< Distributed node counts for fmm calls
    const int n_trg = r_trg.cols();
    const Eigen::MatrixXd node_positions = get_node_positions(); //< Distributed node positions for fmm calls
    const Eigen::MatrixXd node_normals = get_node_normals();     //< Distributed node normals for fmm calls
    const Eigen::MatrixXd null_matrix;                           //< Empty matrix for dummy arguments to kernels

    // Section: Stresslet kernel
    const Eigen::MatrixXd &r_dl = node_positions; //< "double layer" positions for stresslet kernel
    Eigen::MatrixXd f_dl(9, n_nodes);             //< "double layer" "force" for stresslet kernel

    // double layer density is 2 * outer product of normals with density
    for (int node = 0; node < n_nodes; ++node)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                f_dl(i * 3 + j, node) = 2.0 * node_normals(i, node) * densities(j, node);

    Eigen::MatrixXd v_bdy2all =
        (*stresslet_kernel_)(null_matrix, r_dl, r_trg, null_matrix, f_dl).block(1, 0, 3, n_trg) / eta;

    // Section: Oseen kernel
    Eigen::MatrixXd center_positions = get_center_positions(); //< Distributed center positions for FMM calls
    const Eigen::MatrixXd forces = forces_torques.block(0, 0, 3, center_positions.cols());
    const Eigen::MatrixXd torques = forces_torques.block(3, 0, 3, center_positions.cols());
    v_bdy2all += (*oseen_kernel_)(center_positions, null_matrix, r_trg, forces, null_matrix) / eta;

    // Since rotlet isn't handled via an FMM we don't distribute the nodes, but instead each
    // rank gets the body centers and calculates the center->target rotlet
    constexpr bool override_distributed = true;
    center_positions = get_center_positions(override_distributed);
    v_bdy2all += kernels::rotlet(center_positions, r_trg, torques);

    return v_bdy2all;
}

BodyContainer::BodyContainer(toml::array *body_tables, Params &params) {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    if (!body_tables) {
        return;
    }

    // TODO: Make mult_order and max_pts passable fmm parameters
    stresslet_kernel_ = std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>>(new kernels::FMM<stkfmm::Stk3DFMM>(
        8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::PVel, kernels::stokes_pvel_fmm));
    oseen_kernel_ = std::unique_ptr<kernels::FMM<stkfmm::Stk3DFMM>>(new kernels::FMM<stkfmm::Stk3DFMM>(
        8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::Stokes, kernels::stokes_vel_fmm));

    const int n_bodies_tot = body_tables->size();
    const int n_bodies_extra = n_bodies_tot % world_size_;
    if (world_rank_ == 0)
        std::cout << "Reading in " << n_bodies_tot << " bodies.\n";

    for (int i_body = 0; i_body < n_bodies_tot; ++i_body) {
        toml::table *body_table = body_tables->get_as<toml::table>(i_body);
        bodies.emplace_back(Body(body_table, params));

        auto &body = bodies.back();
        if (world_rank_ == 0)
            std::cout << "Body " << i_body << ": " << body.node_weights_.size() << " [ " << body.position_.transpose()
                      << " ]\n";
    }
}
