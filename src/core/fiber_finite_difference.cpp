#include <skelly_sim.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

#include <fiber_finite_difference.hpp>
#include <kernels.hpp>
#include <periphery.hpp>
#include <utils.hpp>

#include <toml.hpp>

/// @file
/// @brief Implement FiberFiniteDifference class and associated functions

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

const std::string FiberFiniteDifference::BC_name[] = {"Force",           "Torque",   "Velocity",
                                                      "AngularVelocity", "Position", "Angle"};

/// @brief FiberFiniteDifference constructor. Duh.
/// This is the preferred way to initialize a fiber.
///
/// @param[in] fiber_table TOML table with associated fiber values
/// @param[in] eta Fluid viscosity
/// @return FiberFiniteDifference object. Cache values are _not_ calculated.
FiberFiniteDifference::FiberFiniteDifference(toml::value &fiber_table, double eta) {
    std::vector<double> x_array = toml::find<std::vector<double>>(fiber_table, "x");
    n_nodes_ = x_array.size() / 3;
    x_ = Eigen::Map<Eigen::MatrixXd>(x_array.data(), 3, n_nodes_);

    init();

    bending_rigidity_ = toml::find<double>(fiber_table, "bending_rigidity");
    radius_ = toml::find_or<double>(fiber_table, "radius", 0.0125);
    length_ = toml::find<double>(fiber_table, "length");
    length_prev_ = toml::find<double>(fiber_table, "length");
    force_scale_ = toml::find_or<double>(fiber_table, "force_scale", 0.0);
    binding_site_.first = toml::find_or<int>(fiber_table, "parent_body", -1);
    binding_site_.second = toml::find_or<int>(fiber_table, "parent_site", -1);
    minus_clamped_ = toml::find_or<bool>(fiber_table, "minus_clamped", false);

    update_constants(eta);
}

/// @brief Update stokeslet for points along fiber
/// Calls kernels::oseen_tensor_direct with FiberFiniteDifference::x_ as source/target
///
/// Updates: FiberFiniteDifference::stokeslet_
/// @param[in] eta fluid viscosity
void FiberFiniteDifference::update_stokeslet(double eta) { stokeslet_ = kernels::oseen_tensor_direct(x_, x_, eta); }

/// @brief Update all of the derivative internal cache variables
///
/// Updates: FiberFiniteDifference::xs_, FiberFiniteDifference::xss_, FiberFiniteDifference::xsss_,
/// FiberFiniteDifference::xssss_
void FiberFiniteDifference::update_derivatives() {
    auto &fib_mats = matrices_.at(n_nodes_);
    xs_ = std::pow(2.0 / length_prev_, 1) * x_ * fib_mats.D_1_0;
    xss_ = std::pow(2.0 / length_prev_, 2) * x_ * fib_mats.D_2_0;
    xsss_ = std::pow(2.0 / length_prev_, 3) * x_ * fib_mats.D_3_0;
    xssss_ = std::pow(2.0 / length_prev_, 4) * x_ * fib_mats.D_4_0;
}

/// @brief Check if fiber is within some threshold distance of the cortex attachment radius
///
/// Updates: FiberFiniteDifference::bc_minus_, FiberFiniteDifference::bc_plus_
/// @param[in] Periphery object
void FiberFiniteDifference::update_boundary_conditions(Periphery &shell, const periphery_binding_t &periphery_binding) {
    bc_minus_ = is_minus_clamped()
                    ? std::make_pair(FiberFiniteDifference::BC::Velocity,
                                     FiberFiniteDifference::BC::AngularVelocity) // Clamped to body
                    : std::make_pair(FiberFiniteDifference::BC::Force, FiberFiniteDifference::BC::Torque); // Free

    double angle = std::acos(x_.col(x_.cols() - 1).normalized()[2]);
    bool near_periphery = (periphery_binding.active) && (angle >= periphery_binding.polar_angle_start) &&
                          (angle <= periphery_binding.polar_angle_end) &&
                          shell.check_collision(x_, periphery_binding.threshold);
    bc_plus_ =
        near_periphery
            ? std::make_pair(FiberFiniteDifference::BC::Velocity, FiberFiniteDifference::BC::Torque) // Hinge at cortex
            : std::make_pair(FiberFiniteDifference::BC::Force, FiberFiniteDifference::BC::Torque);   // Free
    spdlog::get("SkellySim global")
        ->debug("Set BC on FiberFiniteDifference {}: [{}, {}], [{}, {}]", (void *)this, BC_name[bc_minus_.first],
                BC_name[bc_minus_.second], BC_name[bc_plus_.first], BC_name[bc_plus_.second]);
}

/// @brief Updates the linear operator A_ that defines the linear system
///
/// \f[ A * (X^{n+1}, T^{n+1}) = \textrm{RHS} \f]
/// Updates: FiberFiniteDifference::A_
void FiberFiniteDifference::update_linear_operator(double dt, double eta) {
    int n_nodes_up = n_nodes_;
    int n_nodes_down = n_nodes_;

    const FiberFiniteDifference::fib_mat_t &mats = matrices_.at(n_nodes_);
    ArrayXXd D_1 = mats.D_1_0.transpose() * std::pow(2.0 / length_, 1);
    ArrayXXd D_2 = mats.D_2_0.transpose() * std::pow(2.0 / length_, 2);
    ArrayXXd D_3 = mats.D_3_0.transpose() * std::pow(2.0 / length_, 3);
    ArrayXXd D_4 = mats.D_4_0.transpose() * std::pow(2.0 / length_, 4);

    A_.resize(4 * n_nodes_up, 4 * n_nodes_down);
    A_.setZero();
    typedef Eigen::Block<Eigen::MatrixXd> submat_t;
    submat_t A_XX = A_.block(0 * n_nodes_up, 0 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_XY = A_.block(0 * n_nodes_up, 1 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_XZ = A_.block(0 * n_nodes_up, 2 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_XT = A_.block(0 * n_nodes_up, 3 * n_nodes_down, n_nodes_up, n_nodes_down);

    submat_t A_YX = A_.block(1 * n_nodes_up, 0 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_YY = A_.block(1 * n_nodes_up, 1 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_YZ = A_.block(1 * n_nodes_up, 2 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_YT = A_.block(1 * n_nodes_up, 3 * n_nodes_down, n_nodes_up, n_nodes_down);

    submat_t A_ZX = A_.block(2 * n_nodes_up, 0 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_ZY = A_.block(2 * n_nodes_up, 1 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_ZZ = A_.block(2 * n_nodes_up, 2 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_ZT = A_.block(2 * n_nodes_up, 3 * n_nodes_down, n_nodes_up, n_nodes_down);

    submat_t A_TX = A_.block(3 * n_nodes_up, 0 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_TY = A_.block(3 * n_nodes_up, 1 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_TZ = A_.block(3 * n_nodes_up, 2 * n_nodes_down, n_nodes_up, n_nodes_down);
    submat_t A_TT = A_.block(3 * n_nodes_up, 3 * n_nodes_down, n_nodes_up, n_nodes_down);

    ArrayXd I_vec = VectorXd::Ones(n_nodes_);

    ArrayXd xs_x = xs_.row(0);
    ArrayXd xs_y = xs_.row(1);
    ArrayXd xs_z = xs_.row(2);

    ArrayXd xss_x = xss_.row(0);
    ArrayXd xss_y = xss_.row(1);
    ArrayXd xss_z = xss_.row(2);

    ArrayXd xsss_x = xsss_.row(0);
    ArrayXd xsss_y = xsss_.row(1);
    ArrayXd xsss_z = xsss_.row(2);

    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_nodes_up, n_nodes_down);

    A_XX = beta_tstep_ / dt * identity + bending_rigidity_ * c_0_ * (D_4.colwise() * (I_vec + xs_x.pow(2))).matrix() +
           bending_rigidity_ * c_1_ * (D_4.colwise() * (I_vec - xs_x.pow(2))).matrix();
    A_XY = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_x * xs_y)).matrix();
    A_XZ = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_x * xs_z)).matrix();

    A_YX = A_XY;
    A_YY = beta_tstep_ / dt * identity + bending_rigidity_ * c_0_ * (D_4.colwise() * (I_vec + xs_y.pow(2))).matrix() +
           bending_rigidity_ * c_1_ * (D_4.colwise() * (I_vec - xs_y.pow(2))).matrix();
    A_YZ = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_y * xs_z)).matrix();

    A_ZX = A_XZ;
    A_ZY = A_YZ;
    A_ZZ = beta_tstep_ / dt * identity + bending_rigidity_ * c_0_ * (D_4.colwise() * (I_vec + xs_z.pow(2))).matrix() +
           bending_rigidity_ * c_1_ * (D_4.colwise() * (I_vec - xs_z.pow(2))).matrix();

    A_XT = -(c_0_ * 2.0) * (D_1.colwise() * xs_x);
    A_XT += -(c_0_ + c_1_) * xss_x.matrix().asDiagonal();

    A_YT = -(c_0_ * 2.0) * (D_1.colwise() * xs_y);
    A_YT += -(c_0_ + c_1_) * xss_y.matrix().asDiagonal();

    A_ZT = -(c_0_ * 2.0) * (D_1.colwise() * xs_z);
    A_ZT += -(c_0_ + c_1_) * xss_z.matrix().asDiagonal();

    // A_TX = -(self.c_1 + 7.0 * self.c_0) * self.E * (D_4.T * xss[:, 0]).T - 6.0 * self.c_0 * self.E * (
    //         D_3.T * xsss[:, 0]).T - self.penalty_param * (D_1.T * xs[:, 0]).T

    A_TX = -(c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_x).matrix() -
           6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_x).matrix() -
           penalty_param_ * (D_1.colwise() * xs_x).matrix();

    A_TY = -(c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_y).matrix() -
           6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_y).matrix() -
           penalty_param_ * (D_1.colwise() * xs_y).matrix();

    A_TZ = -(c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_z).matrix() -
           6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_z).matrix() -
           penalty_param_ * (D_1.colwise() * xs_z).matrix();

    A_TT = -2.0 * c_0_ * D_2;
    A_TT += (c_0_ + c_1_) * (xss_x.pow(2) + xss_y.pow(2) + xss_z.pow(2)).matrix().asDiagonal();
}

/// @brief Compute the 'right-hand-side' for the linear system with upsampling
///
/// \f[ A * (X^{n+1}, T^{n+1}) = \textrm{RHS} \f]
/// where
/// \f[ \textrm{RHS} = (X^n / dt + \textrm{flow} + \textrm{Mobility} * \textrm{force\_external}, ...) \f]
/// Updates: FiberFiniteDifference::RHS_
/// @param[in] dt timestep size
/// @param[in] flow [ 3 x n_nodes_ ] matrix of flow field sampled at the fiber points
/// @param[in] f_external [ 3 x n_nodes_ ] matrix of external forces on the fiber points
void FiberFiniteDifference::update_RHS(double dt, CMatrixRef &flow, CMatrixRef &f_external) {
    const int np = n_nodes_;
    const auto &mats = matrices_.at(np);
    MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_, 1);

    ArrayXd x_x = x_.block(0, 0, 1, np).transpose().array();
    ArrayXd x_y = x_.block(1, 0, 1, np).transpose().array();
    ArrayXd x_z = x_.block(2, 0, 1, np).transpose().array();

    ArrayXd xs_x = xs_.block(0, 0, 1, np).transpose().array();
    ArrayXd xs_y = xs_.block(1, 0, 1, np).transpose().array();
    ArrayXd xs_z = xs_.block(2, 0, 1, np).transpose().array();

    ArrayXd alpha = mats.alpha;
    ArrayXd s_dot = (1.0 + alpha) * (0.5 * v_growth_);
    ArrayXd I_arr = ArrayXd::Ones(np);
    RHS_.resize(4 * np);
    RHS_.setZero();

    // TODO (GK) : xs should be calculated at x_rhs when polymerization term is added to the rhs
    RHS_.segment(0 * np, np) = x_x / dt + s_dot * xs_x;
    RHS_.segment(1 * np, np) = x_y / dt + s_dot * xs_y;
    RHS_.segment(2 * np, np) = x_z / dt + s_dot * xs_z;
    RHS_.segment(3 * np, np) = -penalty_param_ * VectorXd::Ones(np);

    if (flow.size()) {
        RHS_.segment(0 * np, np) += flow.row(0);
        RHS_.segment(1 * np, np) += flow.row(1);
        RHS_.segment(2 * np, np) += flow.row(2);

        RHS_.segment(3 * np, np) += (xs_x.transpose() * (flow.row(0) * D_1.matrix()).array() +
                                     xs_y.transpose() * (flow.row(1) * D_1.matrix()).array() +
                                     xs_z.transpose() * (flow.row(2) * D_1.matrix()).array())
                                        .matrix();
    }
    if (f_external.size()) {
        ArrayXXd fs = f_external * D_1;
        ArrayXd f_x = f_external.row(0).array();
        ArrayXd f_y = f_external.row(1).array();
        ArrayXd f_z = f_external.row(2).array();

        // clang-format off
        RHS_.segment(0 * np, np).array() +=
            c_0_ * ((I_arr + xs_x * xs_x) * f_x) +
            c_0_ * ((        xs_x * xs_y) * f_y) +
            c_0_ * ((        xs_x * xs_z) * f_z) +
            c_1_ * ((I_arr - xs_x * xs_x) * f_x) +
            c_1_ * ((      - xs_x * xs_y) * f_y) +
            c_1_ * ((      - xs_x * xs_z) * f_z);

        RHS_.segment(1 * np, np).array() +=
            c_0_ * ((        xs_y * xs_x) * f_x) +
            c_0_ * ((I_arr + xs_y * xs_y) * f_y) +
            c_0_ * ((        xs_y * xs_z) * f_z) +
            c_1_ * ((      - xs_y * xs_x) * f_x) +
            c_1_ * ((I_arr - xs_y * xs_y) * f_y) +
            c_1_ * ((      - xs_y * xs_z) * f_z);

        RHS_.segment(2 * np, np).array() +=
            c_0_ * ((        xs_z * xs_x) * f_x) +
            c_0_ * ((        xs_z * xs_y) * f_y) +
            c_0_ * ((I_arr + xs_z * xs_z) * f_z) +
            c_1_ * ((      - xs_z * xs_x) * f_x) +
            c_1_ * ((      - xs_z * xs_y) * f_y) +
            c_1_ * ((I_arr - xs_z * xs_z) * f_z);

        RHS_.segment(3 * np, np).array() +=
            2 * c_0_ * (xs_x.transpose() * fs.row(0) +
                        xs_y.transpose() * fs.row(1) +
                        xs_z.transpose() * fs.row(2));
        RHS_.segment(3 * np, np).array() +=
            (c_0_ - c_1_) * (xss_.row(0).transpose().array() * f_x +
                             xss_.row(1).transpose().array() * f_y +
                             xss_.row(2).transpose().array() * f_z);
        // clang-format on
    }
}

VectorXd FiberFiniteDifference::matvec(CVectorRef x, CMatrixRef v, CVectorRef v_boundary) const {
    auto &mats = matrices_.at(n_nodes_);
    const int np = n_nodes_;
    const int bc_start_i = 4 * np - 14;
    MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_prev_, 1);
    MatrixXd xsDs = (D_1.array().colwise() * xs_.row(0).transpose().array()).transpose();
    MatrixXd ysDs = (D_1.array().colwise() * xs_.row(1).transpose().array()).transpose();
    MatrixXd zsDs = (D_1.array().colwise() * xs_.row(2).transpose().array()).transpose();

    VectorXd vT = VectorXd(np * 4);
    VectorXd v_x = v.row(0).transpose();
    VectorXd v_y = v.row(1).transpose();
    VectorXd v_z = v.row(2).transpose();

    vT.segment(0 * np, np) = v_x;
    vT.segment(1 * np, np) = v_y;
    vT.segment(2 * np, np) = v_z;
    vT.segment(3 * np, np) = xsDs * v_x + ysDs * v_y + zsDs * v_z;

    VectorXd vT_in = VectorXd::Zero(4 * np);
    vT_in.segment(0, bc_start_i) = mats.P_downsample_bc * vT;

    VectorXd xs_vT = VectorXd::Zero(4 * np); // from body attachments
    const int minus_node = 0;
    const int plus_node = np - 1;
    xs_vT(bc_start_i + 3) = v.col(minus_node).dot(xs_.col(minus_node));

    // Body link BC (match velocities of body to fiber minus end)
    VectorXd y_BC = VectorXd::Zero(4 * np);
    if (v_boundary.size() > 0)
        y_BC.segment(bc_start_i + 0, 7) = v_boundary;

    if (bc_plus_.first == FiberFiniteDifference::Velocity)
        xs_vT(bc_start_i + 10) = v.col(plus_node).dot(xs_.col(plus_node));

    return A_ * x - vT_in + xs_vT + y_BC;
}

/// @brief Calculate the force operator cache variable
///
/// Updates: FiberFiniteDifference::force_operator_
void FiberFiniteDifference::update_force_operator() {
    const int np = n_nodes_;
    const auto &mats = matrices_.at(np);

    MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_, 1);
    MatrixXd D_4 = mats.D_4_0 * std::pow(2.0 / length_, 4);

    force_operator_.resize(3 * np, 4 * np);
    force_operator_.setZero();

    for (int i = 0; i < 3; ++i) {
        force_operator_.block(i * np, i * np, np, np) = -bending_rigidity_ * D_4.transpose();

        force_operator_.block(i * np, 3 * np, np, np).diagonal() = xss_.row(i);

        Eigen::MatrixXd t2 = (D_1.array().rowwise() * xs_.row(i).array()).matrix().transpose();
        force_operator_.block(i * np, 3 * np, np, np) += t2;
    }
}

/// @brief Update the preconditioner.
/// Make sure that your FiberFiniteDifference::A_ is current @see FiberFiniteDifference::update_linear_operator
/// Updates: FiberFiniteDifference::A_LU_
void FiberFiniteDifference::update_preconditioner() { A_LU_.compute(A_); }

/// @brief Update linear operator and RHS due to boundary conditions
/// Updates: FiberFiniteDifference::A_, FiberFiniteDifference::RHS_
/// @param[in] dt current timestep size
/// @param[in] v_on_fiber [3 x n_nodes_] matrix of velocities on fiber nodes
/// @param[in] f_on_fiber [3 x n_nodes_] matrix of forces on fiber nodes
void FiberFiniteDifference::apply_bc_rectangular(double dt, CMatrixRef &v_on_fiber, CMatrixRef &f_on_fiber) {
    const int np = n_nodes_;
    const auto &mats = matrices_.at(np);
    MatrixXd D_1 = mats.D_1_0.transpose() * std::pow(2.0 / length_, 1);
    MatrixXd D_2 = mats.D_2_0.transpose() * std::pow(2.0 / length_, 2);
    MatrixXd D_3 = mats.D_3_0.transpose() * std::pow(2.0 / length_, 3);
    MatrixXd D_4 = mats.D_4_0.transpose() * std::pow(2.0 / length_, 4);

    // Downsample A, leaving last 14 rows untouched
    A_.block(0, 0, 4 * np - 14, 4 * np) = mats.P_downsample_bc * A_;

    // Downsampled RHS, with rest of RHS filled in by BC calculations
    RHS_.segment(0, 4 * np - 14) = mats.P_downsample_bc * RHS_;
    Eigen::VectorXd::SegmentReturnType B_RHS = RHS_.segment(4 * np - 14, 14);
    B_RHS.setZero();
    Eigen::Block<Eigen::MatrixXd> B = A_.block(4 * np - 14, 0, 14, 4 * np);
    B.setZero();

    switch (bc_minus_.first) {
    case BC::Velocity: {
        B(0, 0 * np) = beta_tstep_ / dt;
        B(1, 1 * np) = beta_tstep_ / dt;
        B(2, 2 * np) = beta_tstep_ / dt;
        B.block(3, 0 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(0, 0) * D_3.row(0);
        B.block(3, 1 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(1, 0) * D_3.row(0);
        B.block(3, 2 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(2, 0) * D_3.row(0);
        B.block(3, 3 * np, 1, np) = (2.0 * c_0_) * D_1.row(0);

        // FIXME: Tag fibers with BC_minus_vec[2]
        Vector3d BC_minus_vec_0({0.0, 0.0, 0.0});
        B_RHS.segment(0, 3) = x_.col(0) / dt + BC_minus_vec_0;
        B_RHS(3) = 0.0;

        if (v_on_fiber.size())
            B_RHS(3) -= xs_.col(0).dot(v_on_fiber.col(0));
        if (f_on_fiber.size())
            B_RHS(3) -= 2 * c_0_ * xs_.col(0).dot(f_on_fiber.col(0));

        break;
    }
    case BC::Force: {
        B.block(0, 0 * np, 1, np) = bending_rigidity_ * D_3.row(0);
        B(0, 3 * np) = -xs_(0, 0);
        B.block(1, 1 * np, 1, np) = bending_rigidity_ * D_3.row(0);
        B(1, 3 * np) = -xs_(1, 0);
        B.block(2, 2 * np, 1, np) = bending_rigidity_ * D_3.row(0);
        B(2, 3 * np) = -xs_(2, 0);
        B.block(3, 0 * np, 1, np) = -bending_rigidity_ * D_2.row(0) * xss_(0, 0);
        B.block(3, 1 * np, 1, np) = -bending_rigidity_ * D_2.row(0) * xss_(1, 0);
        B.block(3, 2 * np, 1, np) = -bending_rigidity_ * D_2.row(0) * xss_(2, 0);
        B(3, 3 * np) = -1;

        Vector3d BC_start_vec_0{0.0, 0.0, 0.0};
        if (f_on_fiber.size())
            BC_start_vec_0 = f_on_fiber.col(0);

        B_RHS.segment(0, 3) = BC_start_vec_0;
        B_RHS(3) = BC_start_vec_0.dot(xs_.col(0));

        break;
    }
    default: {
        spdlog::critical("Unimplemented BC encountered in first minus end of apply_bc_rectangular [{}, {}]",
                         BC_name[bc_minus_.first], BC_name[bc_minus_.second]);
        throw std::runtime_error("Unimplemented BC error");
    }
    }

    switch (bc_minus_.second) {
    case BC::AngularVelocity: {
        B.block(4, 0, 1, np) = (beta_tstep_ / dt) * D_1.row(0);
        B.block(5, np, 1, np) = (beta_tstep_ / dt) * D_1.row(0);
        B.block(6, 2 * np, 1, np) = (beta_tstep_ / dt) * D_1.row(0);

        // FIXME: Tag fibers with BC_minus_vec[2]
        Vector3d BC_minus_vec_1({0.0, 0.0, 0.0});
        B_RHS.segment(4, 3) = xs_.col(0) / dt + BC_minus_vec_1;

        break;
    }
    case BC::Torque: {
        B.block(4, 0 * np, 1, np) = D_2.row(0);
        B.block(5, 1 * np, 1, np) = D_2.row(0);
        B.block(6, 2 * np, 1, np) = D_2.row(0);

        Vector3d BC_start_vec_1{0.0, 0.0, 0.0};
        B_RHS.segment(4, 3) = BC_start_vec_1;
        break;
    }
    default: {
        spdlog::critical("Unimplemented BC encountered in second minus end of apply_bc_rectangular [{}, {}]",
                         BC_name[bc_minus_.first], BC_name[bc_minus_.second]);
        throw std::runtime_error("Unimplemented BC error");
    }
    }

    switch (bc_plus_.first) {
    case BC::Velocity: {
        int endc = n_nodes_ - 1;
        int endr = D_3.rows() - 1;
        B(7, 1 * np - 1) = beta_tstep_ / dt;
        B(8, 2 * np - 1) = beta_tstep_ / dt;
        B(9, 3 * np - 1) = beta_tstep_ / dt;
        B.block(10, 0 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * D_3.row(endr) * xss_(0, endc);
        B.block(10, 1 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * D_3.row(endr) * xss_(1, endc);
        B.block(10, 2 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * D_3.row(endr) * xss_(2, endc);
        B.block(10, 3 * np, 1, np) = (2.0 * c_0_) * D_1.row(endr);

        // FIXME: Tag fibers with BC_plus_vec[2]
        Vector3d BC_plus_vec_0{0.0, 0.0, 0.0};
        B_RHS.segment(7, 3) = x_.col(endc) / dt + BC_plus_vec_0;
        B_RHS(10) = 0.0;

        if (v_on_fiber.size())
            B_RHS(10) -= xs_.col(endc).dot(v_on_fiber.col(endc));
        if (f_on_fiber.size())
            B_RHS(10) -= 2 * c_0_ * xs_.col(endc).dot(f_on_fiber.col(endc));
        break;
    }
    case BC::Force: {
        int endc = xss_.cols() - 1;
        int endr = D_2.rows() - 1;

        B.block(7, 0, 1, np) = -bending_rigidity_ * D_3.row(endr);
        B(7, 4 * np - 1) = xs_(0, endc);
        B.block(8, np, 1, np) = -bending_rigidity_ * D_3.row(endr);
        B(8, 4 * np - 1) = xs_(1, endc);
        B.block(9, 2 * np, 1, np) = -bending_rigidity_ * D_3.row(endr);
        B(9, 4 * np - 1) = xs_(2, endc);
        B.block(10, 0 * np, 1, np) = bending_rigidity_ * D_2.row(endr) * xss_(0, endc);
        B.block(10, 1 * np, 1, np) = bending_rigidity_ * D_2.row(endr) * xss_(1, endc);
        B.block(10, 2 * np, 1, np) = bending_rigidity_ * D_2.row(endr) * xss_(2, endc);
        B(10, 4 * np - 1) = 1.0;

        Vector3d BC_plus_vec_0 = {0.0, 0.0, 0.0};
        if (f_on_fiber.size())
            BC_plus_vec_0 = f_on_fiber.col(f_on_fiber.cols() - 1);

        B_RHS.segment(7, 3) = BC_plus_vec_0;
        B_RHS(10) = BC_plus_vec_0.dot(xs_.col(xs_.cols() - 1));
        break;
    }
    default: {
        spdlog::critical("Unimplemented BC encountered in first plus end of apply_bc_rectangular [{}, {}]",
                         BC_name[bc_plus_.first], BC_name[bc_plus_.second]);
        throw std::runtime_error("Unimplemented BC error");
    }
    }

    switch (bc_plus_.second) {
    case BC::Torque: {
        B.block(11, 0 * np, 1, np) = D_2.row(D_2.rows() - 1);
        B.block(12, 1 * np, 1, np) = D_2.row(D_2.rows() - 1);
        B.block(13, 2 * np, 1, np) = D_2.row(D_2.rows() - 1);

        // FIXME: Tag fibers with BC_plus_vec[2]
        Vector3d BC_plus_vec_1({0.0, 0.0, 0.0});
        B_RHS.segment(11, 3) = BC_plus_vec_1;
        break;
    }
    default: {
        spdlog::critical("Unimplemented BC encountered in second plus end of apply_bc_rectangular [{}, {}]",
                         BC_name[bc_plus_.first], BC_name[bc_plus_.second]);
        throw std::runtime_error("Unimplemented BC error");
    }
    }
}

/// @brief Helper function to initialize FiberFiniteDifference::matrices_
/// @tparam n_nodes_finite_diff Number of neighboring points to use in finite difference approximation
/// @return map of FiberFiniteDifference::fib_mat_t initialized for various numbers of points, where the key will be
/// FiberFiniteDifference::n_nodes_
std::unordered_map<int, FiberFiniteDifference::fib_mat_t> compute_matrices_finitediff(int n_nodes_finite_diff) {
    std::unordered_map<int, FiberFiniteDifference::fib_mat_t> res;

    for (auto n_nodes : {8, 16, 24, 32, 48, 64, 96, 128}) {
        auto &mats = res[n_nodes];
        mats.alpha = ArrayXd::LinSpaced(n_nodes, -1.0, 1.0);

        auto n_nodes_roots = n_nodes - 4;
        mats.alpha_roots = 2 * (0.5 + ArrayXd::LinSpaced(n_nodes_roots, 0, n_nodes_roots - 1)) / n_nodes_roots - 1;

        auto n_nodes_tension = n_nodes - 2;
        mats.alpha_tension =
            2 * (0.5 + ArrayXd::LinSpaced(n_nodes_tension, 0, n_nodes_tension - 1)) / n_nodes_tension - 1;

        // this is the order of the finite differencing
        // 2nd order scheme: 3 points for 1st der, 4 points for 2nd, 5 points for 3rd, 6 points for 4th
        // 4th order scheme: 5 points for 1st der, 6 points for 2nd, 7 points for 3rd, 8 points for 4th
        // Pre-transpose so can be left multiplied by our point-vectors-as-columns position format
        mats.D_1_0 = utils::finite_diff(mats.alpha, 1, n_nodes_finite_diff + 1).transpose();
        mats.D_2_0 = utils::finite_diff(mats.alpha, 2, n_nodes_finite_diff + 2).transpose();
        mats.D_3_0 = utils::finite_diff(mats.alpha, 3, n_nodes_finite_diff + 3).transpose();
        mats.D_4_0 = utils::finite_diff(mats.alpha, 4, n_nodes_finite_diff + 4).transpose();

        mats.P_X = utils::barycentric_matrix(mats.alpha, mats.alpha_roots);
        mats.P_T = utils::barycentric_matrix(mats.alpha, mats.alpha_tension);

        mats.weights_0 = ArrayXd::Ones(mats.alpha.size()) * 2.0;
        mats.weights_0(0) = 1.0;
        mats.weights_0(mats.weights_0.size() - 1) = 1.0;
        mats.weights_0 /= (n_nodes - 1);

        const int np = n_nodes;
        mats.P_downsample_bc = MatrixXd::Zero(4 * n_nodes - 14, 4 * n_nodes);
        mats.P_downsample_bc.block(0 * (np - 4), 0 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(1 * (np - 4), 1 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(2 * (np - 4), 2 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(3 * (np - 4), 3 * np, np - 2, np) = mats.P_T;
    }
    return res;
}

// FIXME: Make this an input parameter
const std::unordered_map<int, FiberFiniteDifference::fib_mat_t> FiberFiniteDifference::matrices_ =
    compute_matrices_finitediff(4);
