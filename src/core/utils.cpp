#include <limits>
#include <skelly_sim.hpp>

#include <utils.hpp>

#include <cnpy.hpp>

/// @brief Return resampling matrix P_{N,-m}.
/// @param[in] x [N] vector of points x_k.
/// @param[in] y [N-m] vector
/// @return Resampling matrix P_{N, -m}
Eigen::MatrixXd utils::barycentric_matrix(CArrayRef &x, CArrayRef &y) {
    int N = x.size();
    int M = y.size();

    Eigen::ArrayXd w = Eigen::ArrayXd::Ones(N);
    for (int i = 1; i < N; i += 2)
        w(i) = -1.0;
    w(0) = 0.5;
    w(N - 1) = -0.5 * std::pow(-1, N);

    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(M, N);
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
Eigen::MatrixXd utils::finite_diff(CArrayRef &s, int M, int n_s) {
    int N = s.size() - 1;
    Eigen::MatrixXd D_s = Eigen::MatrixXd::Zero(N + 1, N + 1);
    int n_s_half = (n_s - 1) / 2;
    n_s = n_s - 1;

    for (int xi = 0; xi < s.size(); ++xi) {
        const auto &si = s[xi];
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

        CArrayMap x(s.data() + xlow, xhigh - xlow);

        // Computer coefficients of differential matrices
        double c1 = 1.0;
        double c4 = x[0] - si;
        Eigen::MatrixXd c = Eigen::MatrixXd::Zero(n_s + 1, M + 1);
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

/// @brief Collects eigen arrays of potentially varying sizes across MPI ranks and returns them
/// in one large array to root process.
///
/// @param[in] local_vec vector to collect
/// @returns Concatenated vector of local_vec on rank 0, empty vector otherwise
Eigen::VectorXd utils::collect_into_global(CVectorRef &local_vec) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Eigen::VectorXi sizes(size);
    Eigen::VectorXd global_vec;

    const int local_vec_size = local_vec.size();

    MPI_Gather(&local_vec_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    Eigen::VectorXi displs(size + 1);
    if (rank == 0) {
        displs[0] = 0;
        int global_vec_size = 0;
        for (int i = 0; i < size; ++i) {
            displs[i + 1] = displs[i] + sizes[i];
            global_vec_size += sizes[i];
        }
        global_vec.resize(global_vec_size);
    }

    MPI_Gatherv(local_vec.data(), local_vec.size(), MPI_DOUBLE, global_vec.data(), sizes.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return global_vec;
}

Eigen::MatrixXd utils::load_mat(cnpy::npz_t &npz, const char *var) {
    return Eigen::Map<Eigen::ArrayXXd>(npz[var].data<double>(), npz[var].shape[1], npz[var].shape[0])
        .matrix()
        .transpose();
}

Eigen::VectorXd utils::load_vec(cnpy::npz_t &npz, const char *var) {
    return Eigen::Map<Eigen::VectorXd>(npz[var].data<double>(), npz[var].shape[0]);
}
