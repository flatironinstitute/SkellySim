#include <algorithm>
#include <fiber.hpp>
#include <unordered_map>

void Fiber::update_derivatives() { return; }

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
Fiber::matrix_t finite_diff(Fiber::array_t &s, int M, int n_s) {
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

        Eigen::Map<Fiber::array_t> x(s.data() + xlow, xhigh - xlow);

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

template <int num_points_finite_diff>
std::unordered_map<int, Fiber::fib_mat_t> compute_matrices() {
    std::unordered_map<int, Fiber::fib_mat_t> res;
    typedef Fiber::array_t array_t;

    for (auto num_points : {8, 16, 32, 64, 96}) {
        auto &mats = res[num_points];
        mats.alpha = Fiber::array_t::LinSpaced(num_points, -1.0, 1.0);

        auto num_points_roots = num_points - 4;
        auto alpha_roots =
            2 * (0.5 + array_t::LinSpaced(num_points_roots, 0, num_points_roots - 1)) / num_points_roots - 1;

        auto num_points_tension = num_points - 2;
        auto alpha_tension =
            2 * (0.5 + array_t::LinSpaced(num_points_tension, 0, num_points_tension - 1)) / num_points_tension - 1;

        // this is the order of the finite differencing
        // 2nd order scheme: 3 points for 1st der, 4 points for 2nd, 5 points for 3rd, 6 points for 4th
        // 4th order scheme: 5 points for 1st der, 6 points for 2nd, 7 points for 3rd, 8 points for 4th
        mats.D_1_0 = finite_diff(mats.alpha, 1, num_points_finite_diff + 1);
        mats.D_2_0 = finite_diff(mats.alpha, 2, num_points_finite_diff + 1);
        mats.D_3_0 = finite_diff(mats.alpha, 3, num_points_finite_diff + 1);
        mats.D_4_0 = finite_diff(mats.alpha, 4, num_points_finite_diff + 1);
    }
    return res;
}

const std::unordered_map<int, Fiber::fib_mat_t> Fiber::matrices = compute_matrices<4>();
