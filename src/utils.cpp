#include <utils.hpp>
#include <mpi.h>

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
Eigen::MatrixXd utils::finite_diff(const Eigen::Ref<Eigen::ArrayXd> &s, int M, int n_s) {
    int N = s.size() - 1;
    Eigen::MatrixXd D_s = Eigen::MatrixXd::Zero(s.size(), s.size());
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

        Eigen::Map<const Eigen::ArrayXd> x(s.data() + xlow, xhigh - xlow);

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
Eigen::VectorXd utils::collect_into_global(const Eigen::Ref<const Eigen::VectorXd> &local_vec) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Eigen::VectorXi sizes(size);
    Eigen::VectorXd global_vec;

    const int local_vec_size = local_vec.size();

    MPI_Gather(&local_vec_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    Eigen::VectorXi displs(size + 1);
    displs[0] = 0;
    int global_vec_size = 0;
    for (int i = 0; i < size; ++i) {
        displs[i + 1] = displs[i] + sizes[i];
        global_vec_size += sizes[i];
    }
    if (rank == 0)
        global_vec.resize(global_vec_size);

    MPI_Gatherv(local_vec.data(), local_vec.size(), MPI_DOUBLE, global_vec.data(), sizes.data(), displs.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return global_vec;
}
