#ifndef UTILS_HPP
#define UTILS_HPP
#include <Eigen/Dense>

namespace utils {
Eigen::MatrixXd finite_diff(const Eigen::Ref<Eigen::ArrayXd> &s, int M, int n_s);
Eigen::VectorXd collect_into_global(const Eigen::Ref<const Eigen::VectorXd> &local_vec);
};

#endif
