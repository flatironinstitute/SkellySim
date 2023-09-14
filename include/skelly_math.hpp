#ifndef SKELLY_MATH_HPP_
#define SKELLY_MATH_HPP_

#include <skelly_sim.hpp>

namespace skelly_math {

using Eigen::ArrayXd;
using Eigen::ArrayXXd;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

/// @brief Return resampling matrix P_{N,-m}.
/// @param[in] x [N] vector of points x_k.
/// @param[in] y [N-m] vector
/// @return Resampling matrix P_{N, -m}
MatrixXd barycentric_matrix(ArrayRef &x, ArrayRef &y) {
    int N = x.size();
    int M = y.size();

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

}; // namespace skelly_math

#endif
