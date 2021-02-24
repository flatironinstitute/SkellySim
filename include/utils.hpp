#ifndef UTILS_HPP
#define UTILS_HPP

#include <skelly_sim.hpp>

namespace utils {
Eigen::MatrixXd finite_diff(ArrayRef &s, int M, int n_s);
Eigen::VectorXd collect_into_global(VectorRef &local_vec);
}; // namespace utils

#endif
