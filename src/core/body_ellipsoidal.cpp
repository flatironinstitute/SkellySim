#include <skelly_sim.hpp>

#include <body.hpp>
#include <body_deformable.hpp>
#include <body_ellipsoidal.hpp>
#include <body_spherical.hpp>
#include <cnpy.hpp>
#include <kernels.hpp>
#include <parse_util.hpp>
#include <periphery.hpp>
#include <utils.hpp>

void EllipsoidalBody::step(double dt, VectorRef &body_solution) {
    int sol_offset = 3 * n_nodes_;

    velocity_ = body_solution.segment(sol_offset, 3);
    angular_velocity_ = body_solution.segment(sol_offset + 3, 3);
    solution_vec_ = body_solution;

    Eigen::Vector3d x_new = position_ + velocity_ * dt;
    Eigen::Vector3d phi = angular_velocity_ * dt;

    double phi_norm = phi.norm();
    Eigen::Quaterniond orientation_new = orientation_;
    if (phi_norm) {
        double s = std::cos(0.5 * phi_norm);
        Eigen::Vector3d p = std::sin(0.5 * phi_norm) * phi / phi_norm;
        orientation_new = Eigen::Quaterniond(s, p[0], p[1], p[2]) * orientation_;
        std::stringstream ss;
        ss << x_new.transpose();
        spdlog::debug("Moving body {}: [{}]", (void *)this, ss.str());

        place(x_new, orientation_new);
    }
}

Eigen::VectorXd EllipsoidalBody::apply_preconditioner(VectorRef &x) const { return A_LU_.solve(x); }

Eigen::VectorXd EllipsoidalBody::matvec(MatrixRef &v_body, VectorRef &x_body) const {
    Eigen::VectorXd res(get_solution_size());

    CMatrixMap d(x_body.data(), 3, n_nodes_);      // Body 'densities'
    CVectorMap U(x_body.data() + 3 * n_nodes_, 6); // Body velocities

    VectorMap res_nodes(res.data(), n_nodes_ * 3);
    VectorMap res_com(res.data() + n_nodes_ * 3, 6);
    Eigen::VectorXd cx = Eigen::VectorXd::Zero(3 * n_nodes_);
    Eigen::VectorXd cy = Eigen::VectorXd::Zero(3 * n_nodes_);
    Eigen::VectorXd cz = Eigen::VectorXd::Zero(3 * n_nodes_);

    for (int i = 0; i < n_nodes_; ++i) {
        cx.segment(i * 3, 3) += d(0, i) / node_weights_(i) * ex_.col(i);
        cy.segment(i * 3, 3) += d(1, i) / node_weights_(i) * ey_.col(i);
        cz.segment(i * 3, 3) += d(2, i) / node_weights_(i) * ez_.col(i);
    }

    Eigen::VectorXd KU = K_ * U;
    Eigen::VectorXd KTLambda = K_.transpose() * CVectorMap(d.data(), 3 * n_nodes_);

    res_nodes = -(cx + cy + cz) - KU + CVectorMap(v_body.data(), n_nodes_ * 3);
    res_com = -KTLambda + U;
    return res;
}

void EllipsoidalBody::min_copy(const std::shared_ptr<EllipsoidalBody> &other) {
    this->position_ = other->position_;
    this->orientation_ = other->orientation_;
    this->place(this->position_, this->orientation_);
    this->solution_vec_ = other->solution_vec_;
}

/// @brief Update internal EllipsoidalBody::K_ matrix variable
/// @see update_preconditioner
void EllipsoidalBody::update_K_matrix() {
    K_.resize(3 * n_nodes_, 6);
    K_.setZero();
    for (int i = 0; i < n_nodes_; ++i) {
        // J matrix
        K_.block(i * 3, 0, 3, 3).diagonal().array() = 1.0;
        // rot matrix
        Eigen::Vector3d vec = node_positions_.col(i) - position_;
        K_.block(i * 3 + 0, 3, 1, 3) = Eigen::RowVector3d({0.0, vec[2], -vec[1]});
        K_.block(i * 3 + 1, 3, 1, 3) = Eigen::RowVector3d({-vec[2], 0.0, vec[0]});
        K_.block(i * 3 + 2, 3, 1, 3) = Eigen::RowVector3d({vec[1], -vec[0], 0.0});
    }
}

/// @brief Update internal variables that need to be recomputed only once after calling EllipsoidalBody::move
///
/// @see update_singularity_subtraction_vecs
/// @see update_K_matrix
/// @see update_preconditioner
/// @param[in] eta fluid viscosity
void EllipsoidalBody::update_cache_variables(double eta) {
    update_singularity_subtraction_vecs(eta);
    update_K_matrix();
    update_preconditioner(eta);
}

/// @brief Update the preconditioner and associated linear operator
///
/// Updates: EllipsoidalBody::A_, EllipsoidalBody::_LU_
/// @param[in] eta
void EllipsoidalBody::update_preconditioner(double eta) {
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
/// Updates only EllipsoidalBody::RHS_
/// \f[ \textrm{RHS} = -\left[{\bf v}_0, {\bf v}_1, ..., {\bf v}_N \right] \f]
/// @param[in] v_on_body [ 3 x n_nodes ] matrix representing the velocity on each node
void EllipsoidalBody::update_RHS(MatrixRef &v_on_body) {
    RHS_.resize(n_nodes_ * 3 + 6);
    RHS_.segment(0, n_nodes_ * 3) = -CVectorMap(v_on_body.data(), v_on_body.size());
    RHS_.segment(n_nodes_ * 3, 6).setZero();
}

/// @brief Move body to new position with new orientation
///
/// Updates: EllipsoidalBody::position_, EllipsoidalBody::orientation_, EllipsoidalBody::node_positions_,
/// EllipsoidalBody::node_normals_, EllipsoidalBody::nucleation_sites_
/// @param[in] new_pos new lab frame position to move the body centroid
/// @param[in] new_orientation new orientation of the body
void EllipsoidalBody::place(const Eigen::Vector3d &new_pos, const Eigen::Quaterniond &new_orientation) {
    position_ = new_pos;
    orientation_ = new_orientation;

    Eigen::Matrix3d rot = orientation_.toRotationMatrix();
    for (int i = 0; i < n_nodes_; ++i)
        node_positions_.col(i) = position_ + rot * node_positions_ref_.col(i);

    for (int i = 0; i < n_nodes_; ++i)
        node_normals_.col(i) = rot * node_normals_ref_.col(i);

    for (int i = 0; i < nucleation_sites_ref_.cols(); ++i)
        nucleation_sites_.col(i) = position_ + rot * nucleation_sites_ref_.col(i);
}

/// @brief Calculate/cache the internal 'singularity subtraction vectors', used in linear operator application
///
/// Need to call after calling EllipsoidalBody::move.
/// @see update_preconditioner
///
/// Updates: EllipsoidalBody::ex_, EllipsoidalBody::ey_, EllipsoidalBody::ez_
/// @param[in] eta viscosity of fluid
void EllipsoidalBody::update_singularity_subtraction_vecs(double eta) {
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
/// Updates: EllipsoidalBody::node_positions_ref_, EllipsoidalBody::node_normals_ref_, EllipsoidalBody::node_weights_
///   @param[in] precompute_file path to file containing precompute data in npz format (from numpy.save). See associated
///   utility precompute script `utils/make_precompute_data.py`
void EllipsoidalBody::load_precompute_data(const std::string &precompute_file) {
    cnpy::npz_t precomp = cnpy::npz_load(precompute_file);
    auto load_mat = [](cnpy::npz_t &npz, const char *var) {
        return Eigen::Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0]).matrix();
    };

    auto load_vec = [](cnpy::npz_t &npz, const char *var) {
        return VectorMap(npz[var].data<double>(), npz[var].shape[0]);
    };

    node_positions_ = node_positions_ref_ = load_mat(precomp, "node_positions_ref");
    node_normals_ = node_normals_ref_ = load_mat(precomp, "node_normals_ref");
    node_weights_ = load_vec(precomp, "node_weights");

    n_nodes_ = node_positions_.cols();
}

/// @brief Construct body from relevant toml config and system params
///
///   @param[in] body_table toml table from pre-parsed config
///   @param[in] params Pre-constructed Params object
///   surface).
///   @return Body object that has been appropriately rotated. Other internal cache variables are _not_ updated.
/// @see update_cache_variables
EllipsoidalBody::EllipsoidalBody(const toml::value &body_table, const Params &params) : Body(body_table, params) {
    using parse_util::convert_array;
    using namespace parse_util;
    using std::string;
    string precompute_file = toml::find<string>(body_table, "precompute_file");
    load_precompute_data(precompute_file);

    radius_ = toml::find_or<double>(body_table, "radius", 0.0);

    // TODO: add body assertions so that input file and precompute data necessarily agree
    if (body_table.contains("position"))
        position_ = convert_array<>(body_table.at("position").as_array());

    if (body_table.contains("orientation"))
        orientation_ = convert_array<Eigen::Quaterniond>(body_table.at("orientation").as_array());
    else
        orientation_ = orientation_ref_;

    if (body_table.contains("nucleation_sites")) {
        nucleation_sites_ref_ = convert_array<>(body_table.at("nucleation_sites").as_array());
        nucleation_sites_ref_.resize(3, nucleation_sites_ref_.size() / 3);
        nucleation_sites_ = nucleation_sites_ref_;
    }

    if (body_table.contains("external_force"))
        external_force_ = convert_array<>(body_table.at("external_force").as_array());
    if (body_table.contains("external_torque"))
        external_torque_ = convert_array<>(body_table.at("external_torque").as_array());

    // Check if we are doing something like oscillatory force and override what we just read in
    string external_force_type_str = "Linear";
    if (body_table.contains("external_force_type")) {
        external_force_type_str = toml::find<string>(body_table, "external_force_type");
        // external_force_ now sets the vector direction of the force, XXX: check if unit force
        if (external_force_type_str == "Linear") {
            external_force_type_ = EXTFORCE::Linear;
        } else if (external_force_type_str == "Oscillatory") {
            external_force_type_ = EXTFORCE::Oscillatory;
            extforce_oscillation_amplitude_ = toml::find<double>(body_table, "external_oscillation_force_amplitude");
            extforce_oscillation_omega_ =
                2.0 * M_PI * toml::find<double>(body_table, "external_oscillation_force_frequency");
            extforce_oscillation_phase_ = toml::find<double>(body_table, "external_oscillation_force_phase");
        }
    }

    // Print functionality (TODO)
    spdlog::info("  body external force type        = {}", EXTFORCE_name[external_force_type_]);
    spdlog::info("  body external force director    = [ {}, {}, {} ]", external_force_[0], external_force_[1],
                 external_force_[2]);
    if (external_force_type_ == EXTFORCE::Oscillatory) {
        spdlog::info("  body external oscillatory force amplitutde  = {}", extforce_oscillation_amplitude_);
        spdlog::info("  body external oscillatory force frequency   = {}", extforce_oscillation_omega_ / (2.0 * M_PI));
        spdlog::info("  body external oscillatory force phase       = {}", extforce_oscillation_phase_);
    }

    place(position_, orientation_);

    update_cache_variables(params.eta);
}

/// @brief Check for collision with body and periphery.
/// This is a double dispatch routine to handle polymorphism of the callee
///
/// @param[in] periphery Reference to Periphery object to check against
/// @param[in] threshold Minimum between surfaces to consider it a collision
/// @return true if collision detected, false otherwise
bool EllipsoidalBody::check_collision(const Periphery &periphery, double threshold) const {
    return periphery.check_collision(*this, threshold);
}

/// @brief Check for collision with body and another body.
/// This is a double dispatch routine to handle polymorphism of the callee
///
/// @param[in] body Reference to Body object to check against
/// @param[in] threshold Minimum between surfaces to consider it a collision
/// @return true if collision detected, false otherwise
bool EllipsoidalBody::check_collision(const Body &body, double threshold) const {
    return body.check_collision(*this, threshold);
}

/// @brief Check for collision with EllipsoidalBody and another SphericalBody.
/// This routine is the end result of the double dispatch call on a generic Body callee that is actually a
/// EllipsoidalBody.
///
/// @param[in] body Reference to SphericalBody object to check against
/// @param[in] threshold Minimum between surfaces to consider it a collision
/// @return true if collision detected, false otherwise
bool EllipsoidalBody::check_collision(const SphericalBody &body, double threshold) const {
    spdlog::warn("check_collision not defined for EllipsoidalBody->SphericalBody");
    return false;
}

/// @brief Check for collision with EllipsoidalBody another DeformableBody.
/// This routine is the end result of the double dispatch call on a generic Body callee that is actually a
/// EllipsoidalBody.
///
/// @param[in] body Reference to EllipsoidalBody object to check against
/// @param[in] threshold Minimum between surfaces to consider it a collision
/// @return true if collision detected, false otherwise
bool EllipsoidalBody::check_collision(const DeformableBody &body, double threshold) const {
    spdlog::warn("check_collision not defined for EllipsoidalBody->DeformableBody");
    return false;
}

/// @brief Check for collision with EllipsoidalBody another DeformableBody.
/// This routine is the end result of the double dispatch call on a generic Body callee that is actually a
/// EllipsoidalBody.
///
/// @param[in] body Reference to EllipsoidalBody object to check against
/// @param[in] threshold Minimum between surfaces to consider it a collision
/// @return true if collision detected, false otherwise
bool EllipsoidalBody::check_collision(const EllipsoidalBody &body, double threshold) const {
    spdlog::warn("check_collision not defined for EllipsoidalBody->EllipsoidalBody");
    return false;
}
