#include <algorithm>
#include <fiber.hpp>
#include <unordered_map>

Eigen::MatrixX3d oseen_tensor_contract_direct(const Eigen::Ref<Fiber::matrix_t> &r_src,
                                              const Eigen::Ref<Fiber::matrix_t> &r_trg,
                                              const Eigen::Ref<Fiber::matrix_t> &density, double eta = 1.0,
                                              double reg = 5E-3, double epsilon_distance = 1E-5) {

    using namespace Eigen;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;

    // # Compute matrix of size 3N \times 3N
    MatrixX3d res = MatrixXd::Zero(N_trg, 3);

    const double factor = 1.0 / (8.0 * M_PI * eta);
    const double reg2 = std::pow(reg, 2);
    for (int i_trg = 0; i_trg < N_trg; ++i_trg) {
        for (int i_src = 0; i_src < N_src; ++i_src) {
            double fr, gr;
            double dx = r_src(i_src, 0) - r_trg(i_trg, 0);
            double dy = r_src(i_src, 1) - r_trg(i_trg, 1);
            double dz = r_src(i_src, 2) - r_trg(i_trg, 2);
            double dr2 = dx * dx + dy * dy + dz * dz;
            double dr = sqrt(dr2);

            if (dr > epsilon_distance) {
                fr = factor / dr;
                gr = factor / std::pow(dr, 3);
            } else {
                double denom_inv = 1.0 / sqrt(std::pow(dr, 2) + reg2);
                fr = factor * denom_inv;
                gr = factor * std::pow(denom_inv, 3);
            }

            double Mxx = fr + gr * dx * dx;
            double Mxy = gr * dx * dy;
            double Mxz = gr * dx * dz;
            double Myy = fr + gr * dy * dy;
            double Myz = gr * dy * dz;
            double Mzz = fr + gr * dz * dz;

            res(i_trg, 0) += Mxx * density(i_src, 0) + Mxy * density(i_src, 1) + Mxz * density(i_src, 2);
            res(i_trg, 1) += Mxy * density(i_src, 0) + Myy * density(i_src, 1) + Myz * density(i_src, 2);
            res(i_trg, 2) += Mxz * density(i_src, 0) + Myz * density(i_src, 1) + Mzz * density(i_src, 2);
        }
    }

    return res;
}

// Build the Oseen tensor for N points (sources == targets).
// Set to zero diagonal terms.
//
// G = f(r) * I + g(r) * (r.T*r)
//
// Input:
//   r_vectors = coordinates.
//   eta = (default 1.0) viscosity
//   reg = (default 5e-3) regularization term
//   epsilon_distance = (default 1e-10) set elements to zero for distances < epsilon_distance.
//
// Output:
//   G = Oseen tensor with dimensions (3*num_points) x (3*num_points).
Fiber::matrix_t oseen_tensor_direct(const Eigen::Ref<Fiber::matrix_t> &r_src, const Eigen::Ref<Fiber::matrix_t> &r_trg,
                                    double eta = 1.0, double reg = 5E-3, double epsilon_distance = 1E-5) {

    using namespace Eigen;
    const int N_src = r_src.size() / 3;
    const int N_trg = r_trg.size() / 3;

    // # Compute matrix of size 3N \times 3N
    MatrixXd G = MatrixXd::Zero(r_trg.size(), r_src.size());

    const double factor = 1.0 / (8.0 * M_PI * eta);
    const double reg2 = std::pow(reg, 2);
    for (int i_src = 0; i_src < N_trg; ++i_src) {
        for (int i_trg = 0; i_trg < N_src; ++i_trg) {
            double fr, gr;
            double dx = r_src(i_src, 0) - r_trg(i_trg, 0);
            double dy = r_src(i_src, 1) - r_trg(i_trg, 1);
            double dz = r_src(i_src, 2) - r_trg(i_trg, 2);
            double dr2 = dx * dx + dy * dy + dz * dz;
            double dr = sqrt(dr2);

            if (dr > epsilon_distance) {
                fr = factor / dr;
                gr = factor / std::pow(dr, 3);
            } else {
                double denom_inv = 1.0 / sqrt(std::pow(dr, 2) + reg2);
                fr = factor * denom_inv;
                gr = factor * std::pow(denom_inv, 3);
            }

            G(i_trg * 3 + 0, i_src * 3 + 0) = fr + gr * dx * dx;
            G(i_trg * 3 + 0, i_src * 3 + 1) = gr * dx * dy;
            G(i_trg * 3 + 0, i_src * 3 + 2) = gr * dx * dz;

            G(i_trg * 3 + 1, i_src * 3 + 0) = gr * dy * dx;
            G(i_trg * 3 + 1, i_src * 3 + 1) = fr + gr * dy * dy;
            G(i_trg * 3 + 1, i_src * 3 + 2) = gr * dy * dz;

            G(i_trg * 3 + 2, i_src * 3 + 0) = gr * dz * dx;
            G(i_trg * 3 + 2, i_src * 3 + 1) = gr * dz * dy;
            G(i_trg * 3 + 2, i_src * 3 + 2) = fr + gr * dz * dz;
        }
    }

    // ArrayXXd drx(N, N);
    // ArrayXXd dry(N, N);
    // ArrayXXd drz(N, N);

    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         drx(j, i) = r_vectors(i, 0) - r_vectors(j, 0);
    //         dry(j, i) = r_vectors(i, 1) - r_vectors(j, 1);
    //         drz(j, i) = r_vectors(i, 2) - r_vectors(j, 2);
    //     }
    // }
    // ArrayXXd dr = (drx.pow(2) + dry.pow(2) + drz.pow(2)).sqrt();

    // ArrayXXd fr = MatrixXd::Zero(N, N);
    // ArrayXXd gr = MatrixXd::Zero(N, N);
    // {
    //     const double factor = 1.0 / (8.0 * M_PI * eta);
    //     const double reg2 = std::pow(reg, 2);
    //     for (int i = 0; i < N; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             if (dr(j, i) > epsilon_distance) {
    //                 fr(j, i) = factor / dr(j, i);
    //                 gr(j, i) = factor / std::pow(dr(j, i), 3);
    //             } else {
    //                 // Need to regularize when points below threshold
    //                 double denom = sqrt(std::pow(dr(j, i), 2) + reg2);
    //                 fr(j, i) = factor / denom;
    //                 gr(j, i) = factor / std::pow(denom, 3);
    //             }
    //         }
    //     }
    // }

    // constexpr int row_stride = 3;
    // const int col_stride = 9 * N;
    // typedef Stride<Dynamic, row_stride> viewstride;
    // Map<MatrixXd, 0, viewstride>(G.data(), N, N, viewstride(col_stride, row_stride)) = drx + gr * drx * drx;
    // Map<MatrixXd, 0, viewstride>(G.data() + 3 * N, N, N, viewstride(col_stride, row_stride)) = gr * drx * dry;
    // Map<MatrixXd, 0, viewstride>(G.data() + 2 * 3 * N, N, N, viewstride(col_stride, row_stride)) = gr * drx * drz;

    // Map<MatrixXd, 0, viewstride>(G.data() + 1, N, N, viewstride(col_stride, row_stride)) = gr * dry * drx;
    // Map<MatrixXd, 0, viewstride>(G.data() + 1 + 3 * N, N, N, viewstride(col_stride, row_stride)) = fr + gr * dry *
    // dry; Map<MatrixXd, 0, viewstride>(G.data() + 1 + 2 * 3 * N, N, N, viewstride(col_stride, row_stride)) = gr * dry
    // * drz;

    // Map<MatrixXd, 0, viewstride>(G.data() + 2, N, N, viewstride(col_stride, row_stride)) = gr * drz * drx;
    // Map<MatrixXd, 0, viewstride>(G.data() + 2 + 3 * N, N, N, viewstride(col_stride, row_stride)) = gr * drz * dry;
    // Map<MatrixXd, 0, viewstride>(G.data() + 2 + 2 * 3 * N, N, N, viewstride(col_stride, row_stride)) =
    //     fr + gr * drz * drz;

    return G;
}

void Fiber::update_stokeslet() {
    stokeslet = oseen_tensor_direct(x, x);
    for (int i = 0; i < num_points * 3; ++i)
        stokeslet(i, i) = 0.0;
}

void Fiber::update_derivatives() {
    auto &fib_mats = matrices.at(num_points);
    xs = std::pow(2.0 / length, 1) * fib_mats.D_1_0 * x;
    xss = std::pow(2.0 / length, 2) * fib_mats.D_2_0 * x;
    xsss = std::pow(2.0 / length, 3) * fib_mats.D_3_0 * x;
    xssss = std::pow(2.0 / length, 4) * fib_mats.D_4_0 * x;
}

//  Following the paper Calculation of weights in finite different formulas,
//  Bengt Fornberg, SIAM Rev. 40 (3), 685 (1998).
//
//  Inputs:
//  s = grid points
//  M = order of the highest derivative to compute
//  n_s = support to compute the derivatives
//
//  Outputs:
//  D_s = Mth derivative matrix
Fiber::matrix_t finite_diff(const Fiber::array_t &s, int M, int n_s) {
    int N = s.size() - 1;
    Fiber::matrix_t D_s = Fiber::matrix_t::Zero(s.size(), s.size());
    int n_s_half = (n_s - 1) / 2;
    n_s = n_s - 1;

    for (size_t xi = 0; xi < s.size(); ++xi) {
        auto &si = s[xi];
        int xlow, xhigh;

        if (xi < n_s_half) {
            xlow = 0;
            xhigh = n_s + 1;
        } else if (xi > (s.size() - n_s_half - 2)) {
            xlow = -n_s - 1;
            xhigh = s.size();
        } else {
            xlow = xi - n_s_half;
            xhigh = xi - n_s_half + n_s + 1;
        }
        xlow = xlow < 0 ? s.size() + xlow : xlow;

        Eigen::Map<const Fiber::array_t> x(s.data() + xlow, xhigh - xlow);

        // Computer coefficients of differential matrices
        double c1 = 1.0;
        double c4 = x[0] - si;
        Fiber::matrix_t c = Fiber::matrix_t::Zero(n_s + 1, M + 1);
        c(0, 0) = 1.0;

        for (int i = 1; i < n_s + 1; ++i) {
            int mn = std::min(i, M);
            double c2 = 1.0;
            double c5 = c4;
            c4 = x(i) - si;
            for (int j = 0; j < i; ++j) {
                double c3 = x(i) - x(j);
                c2 = c2 * c3;
                if (j == i - 1) {
                    for (int k = mn; k > 0; --k) {
                        c(i, k) = c1 * (k * c(i - 1, k - 1) - c5 * c(i - 1, k)) / c2;
                    }
                    c(i, 0) = -c1 * c5 * c(i - 1, 0) / c2;
                }
                for (int k = mn; k > 0; --k) {
                    c(j, k) = (c4 * c(j, k) - k * c(j, k - 1)) / c3;
                }
                c(j, 0) = c4 * c(j, 0) / c3;
            }
            c1 = c2;
        }

        for (int i = 0; i < n_s + 1; ++i) {
            D_s(xi, xlow + i) = c(i, M);
        }
    }
    return D_s;
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

    for (auto num_points : {8, 16, 32, 64, 96}) {
        auto &mats = res[num_points];
        mats.alpha = Fiber::array_t::LinSpaced(num_points, -1.0, 1.0);

        auto num_points_roots = num_points - 4;
        mats.alpha_roots =
            2 * (0.5 + array_t::LinSpaced(num_points_roots, 0, num_points_roots - 1)) / num_points_roots - 1;

        auto num_points_tension = num_points - 2;
        mats.alpha_tension =
            2 * (0.5 + array_t::LinSpaced(num_points_tension, 0, num_points_tension - 1)) / num_points_tension - 1;

        // this is the order of the finite differencing
        // 2nd order scheme: 3 points for 1st der, 4 points for 2nd, 5 points for 3rd, 6 points for 4th
        // 4th order scheme: 5 points for 1st der, 6 points for 2nd, 7 points for 3rd, 8 points for 4th
        mats.D_1_0 = finite_diff(mats.alpha, 1, num_points_finite_diff + 1);
        mats.D_2_0 = finite_diff(mats.alpha, 2, num_points_finite_diff + 1);
        mats.D_3_0 = finite_diff(mats.alpha, 3, num_points_finite_diff + 1);
        mats.D_4_0 = finite_diff(mats.alpha, 4, num_points_finite_diff + 1);

        mats.P_X = barycentric_matrix(mats.alpha, mats.alpha_roots);
        mats.P_T = barycentric_matrix(mats.alpha, mats.alpha_tension);
        mats.P_cheb_representations_all_dof = Fiber::matrix_t::Zero(4 * num_points - 14, 4 * num_points);

        mats.weights_0 = Fiber::array_t::Ones(mats.alpha.size()) * 2.0;
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

void FiberContainer::update_stokeslets() {
    for (Fiber &fib : fibers)
        fib.update_stokeslet();
}

void FiberContainer::flow(const Eigen::Ref<Eigen::MatrixX3d> r_trg, const Eigen::Ref<Eigen::MatrixX3d> &forces) {
    const size_t n_pts_tot = forces.size() / 3;

    Eigen::MatrixX3d weighted_forces(n_pts_tot, 3);
    Eigen::MatrixX3d r_src(n_pts_tot, 3);
    size_t offset = 0;

    for (const Fiber &fib : fibers) {
        const Fiber::array_t &weights = fib.matrices.at(fib.num_points).weights_0;
        double half_length = 0.5 * fib.length;
        for (int i_pt = 0; i_pt < fib.num_points; ++i_pt) {
            for (int i = 0; i < 3; ++i) {
                weighted_forces(i_pt + offset, i) = half_length * weights[i] * forces(i_pt + offset, i);
                r_src(i_pt + offset, i) = fib.x(i_pt, i);
            }
        }
        offset += fib.num_points;
    }

    Eigen::MatrixX3d vel = oseen_tensor_contract_direct(r_src, r_trg, weighted_forces);
}
