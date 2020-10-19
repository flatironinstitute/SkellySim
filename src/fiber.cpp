#include <algorithm>
#include <fiber.hpp>
#include <iostream>
#include <kernels.hpp>
#include <unordered_map>
#include <utils.hpp>

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
    Eigen::ArrayXXd D_1 = mats.D_1_0.transpose() * std::pow(2.0 / length_, 1);
    Eigen::ArrayXXd D_2 = mats.D_2_0.transpose() * std::pow(2.0 / length_, 2);
    Eigen::ArrayXXd D_3 = mats.D_3_0.transpose() * std::pow(2.0 / length_, 3);
    Eigen::ArrayXXd D_4 = mats.D_4_0.transpose() * std::pow(2.0 / length_, 4);

    A_ = Eigen::MatrixXd::Zero(4 * num_points_up, 4 * num_points_down);
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

    Eigen::VectorXd I_vec = Eigen::VectorXd::Ones(num_points_);

    Eigen::ArrayXd xs_x = xs_.block(0, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xs_y = xs_.block(1, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xs_z = xs_.block(2, 0, 1, num_points_).transpose().array();

    Eigen::ArrayXd xss_x = xss_.block(0, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xss_y = xss_.block(1, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xss_z = xss_.block(2, 0, 1, num_points_).transpose().array();

    Eigen::ArrayXd xsss_x = xsss_.block(0, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xsss_y = xsss_.block(1, 0, 1, num_points_).transpose().array();
    Eigen::ArrayXd xsss_z = xsss_.block(2, 0, 1, num_points_).transpose().array();

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
void Fiber::compute_RHS(double dt, Eigen::Ref<Eigen::MatrixXd> flow, Eigen::Ref<Eigen::MatrixXd> f_external) {
    const int np = num_points_;
    const auto &mats = matrices_.at(np);
    Eigen::MatrixXd D_1 = mats.D_1_0 * std::pow(2.0 / length_, 1);
    Eigen::MatrixXd D_2 = mats.D_2_0 * std::pow(2.0 / length_, 2);
    Eigen::MatrixXd D_3 = mats.D_3_0 * std::pow(2.0 / length_, 3);
    Eigen::MatrixXd D_4 = mats.D_4_0 * std::pow(2.0 / length_, 4);

    Eigen::ArrayXd x_x = x_.block(0, 0, 1, np).transpose().array();
    Eigen::ArrayXd x_y = x_.block(1, 0, 1, np).transpose().array();
    Eigen::ArrayXd x_z = x_.block(2, 0, 1, np).transpose().array();

    Eigen::ArrayXd xs_x = xs_.block(0, 0, 1, np).transpose().array();
    Eigen::ArrayXd xs_y = xs_.block(1, 0, 1, np).transpose().array();
    Eigen::ArrayXd xs_z = xs_.block(2, 0, 1, np).transpose().array();

    Eigen::ArrayXd alpha = mats.alpha;
    Eigen::ArrayXd s = (1.0 + alpha) * (0.5 * v_length_);
    Eigen::ArrayXd I_arr = Eigen::ArrayXd::Ones(np);
    RHS_.resize(4 * np);
    RHS_.setZero();

    // TODO (GK) : xs should be calculated at x_rhs when polymerization term is added to the rhs
    RHS_.segment(0 * np, np) = x_x / dt + s * xs_x;
    RHS_.segment(1 * np, np) = x_y / dt + s * xs_y;
    RHS_.segment(2 * np, np) = x_z / dt + s * xs_z;
    RHS_.segment(3 * np, np) = -penalty_param_ * Eigen::VectorXd::Ones(np);

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
        Eigen::ArrayXXd fs = f_external * D_1;
        Eigen::ArrayXd f_x = f_external.row(0).array();
        Eigen::ArrayXd f_y = f_external.row(1).array();
        Eigen::ArrayXd f_z = f_external.row(2).array();

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
        // clang-format on

        RHS_.segment(3 * np, np).array() +=
            2 * c_0_ * (xs_x.transpose() * fs.row(0) + xs_y.transpose() * fs.row(1) + xs_z.transpose() * fs.row(2));
        RHS_.segment(3 * np, np).array() +=
            (c_0_ - c_1_) * (xss_.row(0).transpose().array() * f_x + xss_.row(1).transpose().array() * f_y +
                             xss_.row(2).transpose().array() * f_z);
    }
}

// Return resampling matrix P_{N,-m}.
// Inputs:
//   x = Eigen array, N points x_k.
//   y = Eigen array, N-m points.
Fiber::matrix_t barycentric_matrix(const Eigen::Ref<Eigen::ArrayXd> &x, const Eigen::Ref<Eigen::ArrayXd> &y) {
    int N = x.size();
    int M = y.size();
    int m = N - M;

    Eigen::ArrayXd w = Eigen::ArrayXd::Ones(N);
    for (int i = 1; i < N; i += 2)
        w(i) = -1.0;
    w(0) = 0.5;
    w(N - 1) = -0.5 * std::pow(-1, N);

    Fiber::matrix_t P = Fiber::matrix_t::Zero(M, N);
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
    typedef Fiber::array_t array_t;

    for (auto num_points : {16, 24, 32, 48, 64, 96}) {
        auto &mats = res[num_points];
        mats.alpha = array_t::LinSpaced(num_points, -1.0, 1.0);

        auto num_points_roots = num_points - 4;
        mats.alpha_roots =
            2 * (0.5 + array_t::LinSpaced(num_points_roots, 0, num_points_roots - 1)) / num_points_roots - 1;

        auto num_points_tension = num_points - 2;
        mats.alpha_tension =
            2 * (0.5 + array_t::LinSpaced(num_points_tension, 0, num_points_tension - 1)) / num_points_tension - 1;

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
        mats.P_cheb_representations_all_dof = Fiber::matrix_t::Zero(4 * num_points - 14, 4 * num_points);

        mats.weights_0 = array_t::Ones(mats.alpha.size()) * 2.0;
        mats.weights_0(0) = 1.0;
        mats.weights_0(mats.weights_0.size() - 1) = 1.0;
        mats.weights_0 /= (num_points - 1);

        for (int i = 0; i < num_points - 4; ++i) {
            for (int j = 0; j < num_points; ++j) {
                mats.P_cheb_representations_all_dof(i + 0 * (num_points - 4), j + 0 * num_points) = mats.P_X(i, j);
                mats.P_cheb_representations_all_dof(i + 1 * (num_points - 4), j + 1 * num_points) = mats.P_X(i, j);
                mats.P_cheb_representations_all_dof(i + 2 * (num_points - 4), j + 2 * num_points) = mats.P_X(i, j);
                mats.P_cheb_representations_all_dof(i + 3 * (num_points - 4), j + 3 * num_points) = mats.P_T(i, j);
            }
        }
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

Eigen::MatrixXd FiberContainer::flow(const Eigen::Ref<Eigen::MatrixXd> &forces) {
    // FIXME: Move fmm object and make more flexible
    static kernels::FMM<stkfmm::Stk3DFMM> fmm(8, 500, stkfmm::PAXIS::NONE, stkfmm::KERNEL::Stokes,
                                              kernels::stokes_vel_fmm);
    const size_t n_pts_tot = forces.cols();

    Eigen::MatrixXd weighted_forces(3, n_pts_tot);
    Eigen::MatrixXd r_src(3, n_pts_tot);
    size_t offset = 0;

    for (const Fiber &fib : fibers) {
        const Fiber::array_t &weights = 0.5 * fib.length_ * fib.matrices_.at(fib.num_points_).weights_0;

        for (int i_pt = 0; i_pt < fib.num_points_; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                weighted_forces(i, i_pt + offset) = weights[i] * forces(i, i_pt + offset);
                r_src(i, i_pt + offset) = fib.x_(i, i_pt);
            }
        }
        offset += fib.num_points_;
    }

    // All-to-all
    // FIXME: MPI not compatible with direct calculation
    Eigen::MatrixXd r_trg = r_src;
    // Eigen::MatrixXd vel = kernels::oseen_tensor_contract_direct(r_src, r_trg, weighted_forces);
    Eigen::MatrixXd vel = fmm(r_src, r_trg, weighted_forces);

    // Subtract self term
    // FIXME: Subtracting self flow only works when system has only fibers
    offset = 0;
    for (const Fiber &fib : fibers) {
        Eigen::Map<Eigen::VectorXd> wf_flat(weighted_forces.data() + offset * 3, fib.num_points_ * 3);
        Eigen::Map<Eigen::VectorXd> vel_flat(vel.data() + offset * 3, fib.num_points_ * 3);
        vel_flat -= fib.stokeslet_ * wf_flat;
        offset += fib.num_points_;
    }

    return vel;
}
