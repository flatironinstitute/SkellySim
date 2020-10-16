#include <algorithm>
#include <fiber.hpp>
#include <iostream>
#include <kernels.hpp>
#include <unordered_map>
#include <utils.hpp>

void Fiber::update_stokeslet(double eta) {
    // FIXME: Remove arguments for stokeslet?
    stokeslet = kernels::oseen_tensor_direct(x, x, eta = eta);
}

void Fiber::update_derivatives() {
    auto &fib_mats = matrices.at(num_points);
    xs = std::pow(2.0 / length, 1) * x * fib_mats.D_1_0;
    xss = std::pow(2.0 / length, 2) * x * fib_mats.D_2_0;
    xsss = std::pow(2.0 / length, 3) * x * fib_mats.D_3_0;
    xssss = std::pow(2.0 / length, 4) * x * fib_mats.D_4_0;
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

const std::unordered_map<int, Fiber::fib_mat_t> Fiber::matrices = compute_matrices<4>();

void FiberContainer::update_derivatives() {
    for (Fiber &fib : fibers)
        fib.update_derivatives();
}

void FiberContainer::update_stokeslets(double eta) {
    // FIXME: Remove default arguments for stokeslets
    for (Fiber &fib : fibers)
        fib.update_stokeslet(eta);
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
        const Fiber::array_t &weights = 0.5 * fib.length * fib.matrices.at(fib.num_points).weights_0;

        for (int i_pt = 0; i_pt < fib.num_points; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                weighted_forces(i, i_pt + offset) = weights[i] * forces(i, i_pt + offset);
                r_src(i, i_pt + offset) = fib.x(i, i_pt);
            }
        }
        offset += fib.num_points;
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
        Eigen::Map<Eigen::VectorXd> wf_flat(weighted_forces.data() + offset * 3, fib.num_points * 3);
        Eigen::Map<Eigen::VectorXd> vel_flat(vel.data() + offset * 3, fib.num_points * 3);
        vel_flat -= fib.stokeslet * wf_flat;
        offset += fib.num_points;
    }

    return vel;
}
