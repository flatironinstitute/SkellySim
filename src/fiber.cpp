#include <algorithm>
#include <fiber.hpp>
#include <iostream>
#include <kernels.hpp>
#include <unordered_map>
#include <utils.hpp>

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

void Fiber::update_stokeslet(double eta) {
    // FIXME: Remove arguments for stokeslet?
    stokeslet_ = kernels::oseen_tensor_direct(x_, x_, eta = eta);
}

void Fiber::update_derivatives() {
    auto &fib_mats = matrices_.at(num_points_);
    xs_ = std::pow(2.0 / length_, 1) * x_ * fib_mats.D_1_0;
    xss_ = std::pow(2.0 / length_, 2) * x_ * fib_mats.D_2_0;
    xsss_ = std::pow(2.0 / length_, 3) * x_ * fib_mats.D_3_0;
    xssss_ = std::pow(2.0 / length_, 4) * x_ * fib_mats.D_4_0;
}

// Calculates the linear operator A_ that define the linear system
// ONLY 1st ORDER, USES PRECOMPUTED AND STORED MATRICES
// A * (X^{n+1}, T^{n+1}) = RHS
void Fiber::form_linear_operator(double dt, double eta) {
    int num_points_up = num_points_;
    int num_points_down = num_points_;

    const auto &mats = matrices_.at(num_points_);
    ArrayXXd D_1 = mats.D_1_0.transpose() * std::pow(2.0 / length_, 1);
    ArrayXXd D_2 = mats.D_2_0.transpose() * std::pow(2.0 / length_, 2);
    ArrayXXd D_3 = mats.D_3_0.transpose() * std::pow(2.0 / length_, 3);
    ArrayXXd D_4 = mats.D_4_0.transpose() * std::pow(2.0 / length_, 4);

    A_ = MatrixXd::Zero(4 * num_points_up, 4 * num_points_down);
    auto A_XX = A_.block(0 * num_points_up, 0 * num_points_down, num_points_up, num_points_down);
    auto A_XY = A_.block(0 * num_points_up, 1 * num_points_down, num_points_up, num_points_down);
    auto A_XZ = A_.block(0 * num_points_up, 2 * num_points_down, num_points_up, num_points_down);
    auto A_XT = A_.block(0 * num_points_up, 3 * num_points_down, num_points_up, num_points_down);

    auto A_YX = A_.block(1 * num_points_up, 0 * num_points_down, num_points_up, num_points_down);
    auto A_YY = A_.block(1 * num_points_up, 1 * num_points_down, num_points_up, num_points_down);
    auto A_YZ = A_.block(1 * num_points_up, 2 * num_points_down, num_points_up, num_points_down);
    auto A_YT = A_.block(1 * num_points_up, 3 * num_points_down, num_points_up, num_points_down);

    auto A_ZX = A_.block(2 * num_points_up, 0 * num_points_down, num_points_up, num_points_down);
    auto A_ZY = A_.block(2 * num_points_up, 1 * num_points_down, num_points_up, num_points_down);
    auto A_ZZ = A_.block(2 * num_points_up, 2 * num_points_down, num_points_up, num_points_down);
    auto A_ZT = A_.block(2 * num_points_up, 3 * num_points_down, num_points_up, num_points_down);

    auto A_TX = A_.block(3 * num_points_up, 0 * num_points_down, num_points_up, num_points_down);
    auto A_TY = A_.block(3 * num_points_up, 1 * num_points_down, num_points_up, num_points_down);
    auto A_TZ = A_.block(3 * num_points_up, 2 * num_points_down, num_points_up, num_points_down);
    auto A_TT = A_.block(3 * num_points_up, 3 * num_points_down, num_points_up, num_points_down);

    VectorXd I_vec = VectorXd::Ones(num_points_);

    ArrayXd xs_x = xs_.block(0, 0, 1, num_points_).transpose().array();
    ArrayXd xs_y = xs_.block(1, 0, 1, num_points_).transpose().array();
    ArrayXd xs_z = xs_.block(2, 0, 1, num_points_).transpose().array();

    ArrayXd xss_x = xss_.block(0, 0, 1, num_points_).transpose().array();
    ArrayXd xss_y = xss_.block(1, 0, 1, num_points_).transpose().array();
    ArrayXd xss_z = xss_.block(2, 0, 1, num_points_).transpose().array();

    ArrayXd xsss_x = xsss_.block(0, 0, 1, num_points_).transpose().array();
    ArrayXd xsss_y = xsss_.block(1, 0, 1, num_points_).transpose().array();
    ArrayXd xsss_z = xsss_.block(2, 0, 1, num_points_).transpose().array();

    A_XX = beta_tstep_ / dt * I_vec.asDiagonal();
    A_XX += bending_rigidity_ * (c_0_) * (D_4.colwise() * (I_vec.array() + xs_x.pow(2))).matrix();
    A_XX += bending_rigidity_ * (c_1_) * (D_4.colwise() * (I_vec.array() - xs_x.pow(2))).matrix();
    A_XY = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_x * xs_y)).matrix();
    A_XZ = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_x * xs_z)).matrix();

    A_YX = A_XY;
    A_YY = beta_tstep_ / dt * I_vec.asDiagonal();
    A_YY += bending_rigidity_ * (c_0_) * (D_4.colwise() * (I_vec.array() + xs_y.pow(2))).matrix();
    A_YY += bending_rigidity_ * (c_1_) * (D_4.colwise() * (I_vec.array() - xs_y.pow(2))).matrix();
    A_YZ = bending_rigidity_ * (c_0_ - c_1_) * (D_4.colwise() * (xs_y * xs_z)).matrix();

    A_ZX = A_XZ;
    A_ZY = A_YZ;
    A_ZZ = beta_tstep_ / dt * I_vec.asDiagonal();
    A_ZZ += bending_rigidity_ * (c_0_) * (D_4.colwise() * (I_vec.array() + xs_z.pow(2))).matrix();
    A_ZZ += bending_rigidity_ * (c_1_) * (D_4.colwise() * (I_vec.array() - xs_z.pow(2))).matrix();

    A_XT = -(c_0_ * 2.0) * (D_1.colwise() * xs_x);
    A_XT += -(c_0_ + c_1_) * xss_x.matrix().asDiagonal();

    A_YT = -(c_0_ * 2.0) * (D_1.colwise() * xs_y);
    A_YT += -(c_0_ * 2.0 + c_1_) * xss_y.matrix().asDiagonal();

    A_ZT = -(c_0_ * 2.0) * (D_1.colwise() * xs_z);
    A_ZT += -(c_0_ * 2.0 + c_1_) * xss_z.matrix().asDiagonal();

    A_TX = (-c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_x);
    A_TX -= 6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_x).matrix();
    A_TX -= penalty_param_ * (D_1.colwise() * xs_x).matrix();

    A_TY = (-c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_y);
    A_TY -= 6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_y).matrix();
    A_TY -= penalty_param_ * (D_1.colwise() * xs_y).matrix();

    A_TZ = (-c_1_ + 7.0 * c_0_) * bending_rigidity_ * (D_4.colwise() * xss_z);
    A_TZ -= 6.0 * c_0_ * bending_rigidity_ * (D_3.colwise() * xsss_z).matrix();
    A_TZ -= penalty_param_ * (D_1.colwise() * xs_z).matrix();

    A_TT = -2.0 * c_0_ * D_2;
    A_TT += (c_0_ + c_1_) * (xss_x.pow(2) + xss_y.pow(2) + xss_z.pow(2)).matrix().asDiagonal();

    // FIXME: Add nonlocal interactions to fibers?
}

// Compute the Right Hand Side for the linear system with upsampling
// A * (X^{n+1}, T^{n+1}) = RHS
// with
// RHS = (X^n / dt + flow + Mobility * force_external, ...)
void Fiber::compute_RHS(double dt, const Eigen::Ref<const MatrixXd> flow, const Eigen::Ref<const MatrixXd> f_external) {
    const int np = num_points_;
    const auto &mats = matrices_.at(np);
    MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_, 1);
    MatrixXd D_2 = mats.D_2_0 * std::pow(2.0 / length_, 2);
    MatrixXd D_3 = mats.D_3_0 * std::pow(2.0 / length_, 3);
    MatrixXd D_4 = mats.D_4_0 * std::pow(2.0 / length_, 4);

    ArrayXd x_x = x_.block(0, 0, 1, np).transpose().array();
    ArrayXd x_y = x_.block(1, 0, 1, np).transpose().array();
    ArrayXd x_z = x_.block(2, 0, 1, np).transpose().array();

    ArrayXd xs_x = xs_.block(0, 0, 1, np).transpose().array();
    ArrayXd xs_y = xs_.block(1, 0, 1, np).transpose().array();
    ArrayXd xs_z = xs_.block(2, 0, 1, np).transpose().array();

    ArrayXd alpha = mats.alpha;
    ArrayXd s = (1.0 + alpha) * (0.5 * v_length_);
    ArrayXd I_arr = ArrayXd::Ones(np);
    RHS_.resize(4 * np);
    RHS_.setZero();

    // TODO (GK) : xs should be calculated at x_rhs when polymerization term is added to the rhs
    RHS_.segment(0 * np, np) = x_x / dt + s * xs_x;
    RHS_.segment(1 * np, np) = x_y / dt + s * xs_y;
    RHS_.segment(2 * np, np) = x_z / dt + s * xs_z;
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

void Fiber::form_force_operator() {
    const int np = num_points_;
    const auto &mats = matrices_.at(np);

    MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_, 1);
    MatrixXd D_4 = mats.D_4_0 * std::pow(2.0 / length_, 4);

    force_operator_.resize(3 * np, 4 * np);
    force_operator_.setZero();

    for (int i = 0; i < 3; ++i) {
        force_operator_.block(i * np, i * np, np, np) = -bending_rigidity_ * D_4.transpose();

        force_operator_.block(i * np, 3 * np, np, np).diagonal() = xss_.row(i);

        force_operator_.block(i * np, 3 * np, np, np) +=
            (D_1.array().colwise() * xs_.row(i).transpose().array()).matrix().transpose();
    }
}

void Fiber::build_preconditioner() { A_LU_.compute(A_); }

void Fiber::apply_bc_rectangular(double dt, const Eigen::Ref<const MatrixXd> &v_on_fiber,
                                 const Eigen::Ref<const MatrixXd> &f_on_fiber) {
    const int np = num_points_;
    const auto &mats = matrices_.at(np);
    MatrixXd D_1 = mats.D_1_0.transpose() * std::pow(2.0 / length_, 1);
    MatrixXd D_2 = mats.D_2_0.transpose() * std::pow(2.0 / length_, 2);
    MatrixXd D_3 = mats.D_3_0.transpose() * std::pow(2.0 / length_, 3);
    MatrixXd D_4 = mats.D_4_0.transpose() * std::pow(2.0 / length_, 4);
    auto &x_rhs = x_;
    auto &xs_rhs = xs_;

    // Downsample A, leaving last 14 rows untouched
    A_.block(0, 0, 4 * np - 14, 4 * np) = mats.P_downsample_bc * A_;

    // Downsampled RHS, with rest of RHS filled in by BC calculations
    RHS_.segment(0, 4 * np - 14) = mats.P_downsample_bc * RHS_;
    auto B_RHS = RHS_.segment(4 * np - 14, 14);
    B_RHS.setZero();
    auto B = A_.block(4 * np - 14, 0, 14, 4 * np);
    B.setZero();

    switch (bc_minus_.first) {
    case BC::Velocity:
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
    default:
        std::cerr << "Unimplemented BC encountered in apply_bc_rectangular\n";
        exit(1);
    }

    switch (bc_minus_.second) {
    case BC::AngularVelocity:
        B.block(4, 0, 1, np) = (beta_tstep_ / dt) * D_1.row(0);
        B.block(5, np, 1, np) = (beta_tstep_ / dt) * D_1.row(0);
        B.block(6, 2 * np, 1, np) = (beta_tstep_ / dt) * D_1.row(0);

        // FIXME: Tag fibers with BC_minus_vec[2]
        Vector3d BC_minus_vec_1({0.0, 0.0, 0.0});
        B_RHS.segment(4, 3) = xs_.col(0) / dt + BC_minus_vec_1;

        break;
    default:
        std::cerr << "Unimplemented BC encountered in apply_bc_rectangular\n";
        exit(1);
    }

    switch (bc_plus_.first) {
    // FIXME: implement more BC
    // case BC::Velocity:
    //     B(7, 4 * np - 1) = beta_tstep_ / dt;
    //     B(8, 4 * np - 1) = beta_tstep_ / dt;
    //     B(9, 4 * np - 1) = beta_tstep_ / dt;
    //     int endc = xss_.cols() - 1;
    //     int endr = D_3.rows() - 1;
    //     B.block(10, 0 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(0, endc) * D_3.row(endr);
    //     B.block(10, 1 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(1, endc) * D_3.row(endr);
    //     B.block(10, 2 * np, 1, np) = (6.0 * bending_rigidity_ * c_0_) * xss_(2, endc) * D_3.row(endr);
    //     B.block(10, 3 * np, 1, np) = (2.0 * c_0_) * D_1.row(endr);

    //     // FIXME: Tag fibers with BC_plus_vec[2]
    //     Vector3d BC_plus_vec_0({0.0, 0.0, 0.0});
    //     B_RHS.segment(7, 3) = x_.col(0) / dt + BC_plus_vec_0;
    //     B_RHS(10) = 0.0;

    //     if (v_on_fiber.size())
    //         B_RHS(10) -= xs_.col(xs_.cols() - 1).dot(v_on_fiber.col(v_on_fiber.cols() - 1));
    //     if (f_on_fiber.size())
    //         B_RHS(10) -= 2 * c_0_ * xs_.col(xs_.cols() - 1).dot(f_on_fiber.col(f_on_fiber.cols() - 1));
    //     break;
    case BC::Force:
        B.block(7, 0, 1, np) = -bending_rigidity_ * D_3.row(D_3.rows() - 1);
        B(7, 4 * np - 1) = xs_(0, xs_.cols() - 1);
        B.block(8, np, 1, np) = -bending_rigidity_ * D_3.row(D_3.rows() - 1);
        B(8, 4 * np - 1) = xs_(1, xs_.cols() - 1);
        B.block(9, 2 * np, 1, np) = -bending_rigidity_ * D_3.row(D_3.rows() - 1);
        B(9, 4 * np - 1) = xs_(2, xs_.cols() - 1);
        B.block(10, 0 * np, 1, np) = bending_rigidity_ * D_2.row(D_2.rows() - 1) * xss_(0, xss_.rows() - 1);
        B.block(10, 1 * np, 1, np) = bending_rigidity_ * D_2.row(D_2.rows() - 1) * xss_(1, xss_.rows() - 1);
        B.block(10, 2 * np, 1, np) = bending_rigidity_ * D_2.row(D_2.rows() - 1) * xss_(2, xss_.rows() - 1);
        B(10, 4 * np - 1) = 1.0;

        Vector3d BC_plus_vec_0 = {0.0, 0.0, 0.0};
        if (f_on_fiber.size())
            BC_plus_vec_0 = f_on_fiber.col(f_on_fiber.cols() - 1);

        B_RHS.segment(7, 3) = BC_plus_vec_0;
        B_RHS(10) = BC_plus_vec_0.dot(xs_.col(xs_.cols() - 1));
        break;
    default:
        std::cerr << "Unimplemented BC encountered in apply_bc_rectangular\n";
        exit(1);
    }

    switch (bc_plus_.second) {
    case BC::Torque:
        B.block(11, 0 * np, 1, np) = D_2.row(D_2.rows() - 1);
        B.block(12, 1 * np, 1, np) = D_2.row(D_2.rows() - 1);
        B.block(13, 2 * np, 1, np) = D_2.row(D_2.rows() - 1);

        // FIXME: Tag fibers with BC_plus_vec[2]
        Vector3d BC_plus_vec_1({0.0, 0.0, 0.0});
        B_RHS.segment(11, 3) = BC_plus_vec_1;
        break;
    default:
        std::cerr << "Unimplemented BC encountered in apply_bc_rectangular\n";
        exit(1);
    }
}

// Return resampling matrix P_{N,-m}.
// Inputs:
//   x = Eigen array, N points x_k.
//   y = Eigen array, N-m points.
MatrixXd barycentric_matrix(const Eigen::Ref<const ArrayXd> &x, const Eigen::Ref<const ArrayXd> &y) {
    int N = x.size();
    int M = y.size();
    int m = N - M;

    ArrayXd w = ArrayXd::Ones(N);
    for (int i = 1; i < N; i += 2)
        w(i) = -1.0;
    w(0) = 0.5;
    w(N - 1) = -0.5 * std::pow(-1, N);

    MatrixXd P = MatrixXd::Zero(M, N);
    for (int j = 0; j < M; ++j) {
        double S = 0.0;
        for (int k = 0; k < N; ++k) {
            S += w(k) / (y(j) - x(k));
        }
        for (int k = 0; k < N; ++k) {
            if (std::fabs(y(j) - x(k)) > std::numeric_limits<double>::epsilon())
                P(j, k) = w(k) / (y(j) - x(k)) / S;
            else
                P(j, k) = 1.0;
        }
    }
    return P;
}

template <int num_points_finite_diff>
std::unordered_map<int, Fiber::fib_mat_t> compute_matrices() {
    std::unordered_map<int, Fiber::fib_mat_t> res;

    for (auto num_points : {8, 16, 24, 32, 48, 64, 96}) {
        auto &mats = res[num_points];
        mats.alpha = ArrayXd::LinSpaced(num_points, -1.0, 1.0);

        auto num_points_roots = num_points - 4;
        mats.alpha_roots =
            2 * (0.5 + ArrayXd::LinSpaced(num_points_roots, 0, num_points_roots - 1)) / num_points_roots - 1;

        auto num_points_tension = num_points - 2;
        mats.alpha_tension =
            2 * (0.5 + ArrayXd::LinSpaced(num_points_tension, 0, num_points_tension - 1)) / num_points_tension - 1;

        // this is the order of the finite differencing
        // 2nd order scheme: 3 points for 1st der, 4 points for 2nd, 5 points for 3rd, 6 points for 4th
        // 4th order scheme: 5 points for 1st der, 6 points for 2nd, 7 points for 3rd, 8 points for 4th
        // Pre-transpose so can be left multiplied by our point-vectors-as-columns position format
        mats.D_1_0 = utils::finite_diff(mats.alpha, 1, num_points_finite_diff + 1).transpose();
        mats.D_2_0 = utils::finite_diff(mats.alpha, 2, num_points_finite_diff + 2).transpose();
        mats.D_3_0 = utils::finite_diff(mats.alpha, 3, num_points_finite_diff + 3).transpose();
        mats.D_4_0 = utils::finite_diff(mats.alpha, 4, num_points_finite_diff + 4).transpose();

        mats.P_X = barycentric_matrix(mats.alpha, mats.alpha_roots);
        mats.P_T = barycentric_matrix(mats.alpha, mats.alpha_tension);

        mats.weights_0 = ArrayXd::Ones(mats.alpha.size()) * 2.0;
        mats.weights_0(0) = 1.0;
        mats.weights_0(mats.weights_0.size() - 1) = 1.0;
        mats.weights_0 /= (num_points - 1);

        const int np = num_points;
        mats.P_downsample_bc = MatrixXd::Zero(4 * num_points - 14, 4 * num_points);
        mats.P_downsample_bc.block(0 * (np - 4), 0 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(1 * (np - 4), 1 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(2 * (np - 4), 2 * np, np - 4, np) = mats.P_X;
        mats.P_downsample_bc.block(3 * (np - 4), 3 * np, np - 2, np) = mats.P_T;
    }
    return res;
}

const std::unordered_map<int, Fiber::fib_mat_t> Fiber::matrices_ = compute_matrices<4>();

void FiberContainer::update_derivatives() {
    for (Fiber &fib : fibers)
        fib.update_derivatives();
}

void FiberContainer::update_stokeslets(double eta) {
    // FIXME: Remove default arguments for stokeslets
    for (Fiber &fib : fibers)
        fib.update_stokeslet(eta);
}

void FiberContainer::form_linear_operators(double dt, double eta) {
    for (Fiber &fib : fibers)
        fib.form_linear_operator(dt, eta);
}

VectorXd FiberContainer::apply_preconditioner(const Eigen::Ref<const VectorXd> &x_all) const {
    VectorXd y(x_all.size());
    size_t offset = 0;
    for (auto &fib : fibers) {
        y.segment(offset, 4 * fib.num_points_) = fib.A_LU_.solve(x_all.segment(offset, 4 * fib.num_points_));
        offset += 4 * fib.num_points_;
    }
    return y;
}

VectorXd FiberContainer::matvec(const Eigen::Ref<const VectorXd> &x_all,
                                const Eigen::Ref<const MatrixXd> &v_fib) const {
    int num_points_tot = 0;
    for (auto &fib : fibers)
        num_points_tot += fib.num_points_;

    VectorXd res = VectorXd::Zero(num_points_tot * 4);

    size_t offset = 0;
    for (auto &fib : fibers) {
        auto &mats = fib.matrices_.at(fib.num_points_);
        const int np = fib.num_points_;
        MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / fib.length_, 1);
        MatrixXd xsDs = (D_1.array().colwise() * fib.xs_.row(0).transpose().array()).transpose();
        MatrixXd ysDs = (D_1.array().colwise() * fib.xs_.row(1).transpose().array()).transpose();
        MatrixXd zsDs = (D_1.array().colwise() * fib.xs_.row(2).transpose().array()).transpose();
        VectorXd vT = VectorXd::Zero(np * 4);

        auto v_fib_x = v_fib.row(0).segment(offset, np).transpose();
        auto v_fib_y = v_fib.row(1).segment(offset, np).transpose();
        auto v_fib_z = v_fib.row(2).segment(offset, np).transpose();
        vT.segment(0 * np, np) = v_fib_x;
        vT.segment(1 * np, np) = v_fib_y;
        vT.segment(2 * np, np) = v_fib_z;

        vT.segment(3 * np, np) = xsDs * v_fib_x + ysDs * v_fib_y + zsDs * v_fib_z;

        VectorXd vT_in = VectorXd::Zero(4 * np);
        vT_in.segment(0, 4 * np - 14) = mats.P_downsample_bc * vT;

        VectorXd xs_vT = VectorXd::Zero(4 * np); // from body attachments
        // FIXME: Flow assumes no bodies, only gets BC from minus end magically
        xs_vT(4 * np - 11) = v_fib_x(0) * fib.xs_(0, 0) + v_fib_y(0) * fib.xs_(1, 0) + v_fib_z(0) * fib.xs_(2, 0);
        VectorXd y_BC = VectorXd::Zero(4 * np); // from bodies

        res.segment(4 * offset, 4 * np) = fib.A_ * x_all.segment(4 * offset, 4 * np) - vT_in + xs_vT + y_BC;

        offset += np;
    }

    return res;
}

MatrixXd FiberContainer::get_r_vectors() const {
    const int n_pts_tot = get_total_fib_points();
    MatrixXd r(3, n_pts_tot);
    size_t offset = 0;

    for (const Fiber &fib : fibers) {
        for (int i_pt = 0; i_pt < fib.num_points_; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                r(i, i_pt + offset) = fib.x_(i, i_pt);
            }
        }
        offset += fib.num_points_;
    }

    return r;
}

MatrixXd FiberContainer::flow(const Eigen::Ref<const MatrixXd> &fib_forces,
                              const Eigen::Ref<const MatrixXd> &r_trg_external, double eta) const {
    // FIXME: Move fmm object and make more flexible
    static kernels::FMM<stkfmm::Stk3DFMM> fmm(8, 2000, stkfmm::PAXIS::NONE, stkfmm::KERNEL::Stokes,
                                              kernels::stokes_vel_fmm);
    const size_t n_src = fib_forces.cols();
    const size_t n_trg_external = r_trg_external.cols();

    MatrixXd weighted_forces(3, n_src);
    MatrixXd r_src(3, n_src);
    size_t offset = 0;

    for (const Fiber &fib : fibers) {
        const ArrayXd &weights = 0.5 * fib.length_ * fib.matrices_.at(fib.num_points_).weights_0;

        for (int i_pt = 0; i_pt < fib.num_points_; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                weighted_forces(i, i_pt + offset) = weights(i_pt) * fib_forces(i, i_pt + offset);
                r_src(i, i_pt + offset) = fib.x_(i, i_pt);
            }
        }
        offset += fib.num_points_;
    }

    // All-to-all
    // FIXME: MPI not compatible with direct calculation
    MatrixXd r_trg(3, n_src + n_trg_external);
    r_trg.block(0, 0, 3, n_src) = r_src;
    if (n_trg_external)
        r_trg.block(0, n_src, 3, n_trg_external) = r_trg_external;
    MatrixXd r_dl_dummy, f_dl_dummy;
    MatrixXd vel = fmm(r_src, r_dl_dummy, r_trg, weighted_forces, f_dl_dummy) / eta;

    // Subtract self term
    // FIXME: Subtracting self flow only works when system has only fibers
    offset = 0;
    for (const Fiber &fib : fibers) {
        Eigen::Map<VectorXd> wf_flat(weighted_forces.data() + offset * 3, fib.num_points_ * 3);
        Eigen::Map<VectorXd> vel_flat(vel.data() + offset * 3, fib.num_points_ * 3);
        vel_flat -= fib.stokeslet_ * wf_flat;
        offset += fib.num_points_;
    }

    return vel;
}

MatrixXd FiberContainer::generate_constant_force(double force_scale) const {
    const int n_fib_pts = this->get_total_fib_points();
    MatrixXd f(3, n_fib_pts);
    size_t offset = 0;
    for (const auto &fib : fibers) {
        f.block(0, offset, 3, fib.num_points_) = force_scale * fib.xs_;
        offset += fib.num_points_;
    }
    return f;
}

MatrixXd FiberContainer::apply_fiber_force(const Eigen::Ref<const VectorXd> &x_all) const {
    MatrixXd fw(3, x_all.size() / 4);

    size_t offset = 0;
    for (size_t ifib = 0; ifib < fibers.size(); ++ifib) {
        const auto &fib = fibers[ifib];
        const int np = fib.num_points_;
        auto force_fibers = fib.force_operator_ * x_all.segment(offset * 4, np * 4);
        fw.block(0, offset, 1, np) = force_fibers.segment(0 * np, np).transpose();
        fw.block(1, offset, 1, np) = force_fibers.segment(1 * np, np).transpose();
        fw.block(2, offset, 1, np) = force_fibers.segment(2 * np, np).transpose();

        offset += np;
    }

    return fw;
}

FiberContainer::FiberContainer(std::string fiber_file, double stall_force, double eta) {
    int rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::ifstream ifs(fiber_file);
    std::string token;
    getline(ifs, token);
    const int n_fibs_tot = atoi(token.c_str());
    const int n_fibs_extra = n_fibs_tot % world_size;
    if (rank == 0)
        std::cout << "Reading in " << n_fibs_tot << " fibers.\n";


    std::vector<int> displs(world_size + 1);
    for (int i = 1; i < world_size + 1; ++i) {
        displs[i] = displs[i - 1] + n_fibs_tot / world_size;
        if (i <= n_fibs_extra)
            displs[i]++;
    }

    for (int i_fib = 0; i_fib < n_fibs_tot; ++i_fib) {
        const int i_fib_low = displs[rank];
        const int i_fib_high = displs[rank + 1];
        std::string line;
        getline(ifs, line);
        std::stringstream linestream(line);

        getline(linestream, token, ' ');
        int n_pts = atoi(token.c_str());

        getline(linestream, token, ' ');
        double E = atof(token.c_str());

        getline(linestream, token, ' ');
        double L = atof(token.c_str());

        MatrixXd x(3, n_pts);
        for (int i_pnt = 0; i_pnt < n_pts; ++i_pnt) {
            getline(ifs, line);
            std::stringstream linestream(line);

            if (i_fib >= i_fib_low && i_fib < i_fib_high) {
                for (int i = 0; i < 3; ++i) {
                    getline(linestream, token, ' ');
                    x(i, i_pnt) = atof(token.c_str());
                }
            }
        }

        if (i_fib >= i_fib_low && i_fib < i_fib_high) {
            std::cout << "Fiber " << i_fib << ": " << n_pts << " " << E << " " << L << std::endl;
            fibers.push_back(Fiber(n_pts, E, stall_force, eta));
            auto &fib = fibers.back();

            fib.x_ = x;
            fib.length_ = L;
        }
    }
}
