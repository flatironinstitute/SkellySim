#ifndef UTILS_HPP
#define UTILS_HPP
#include <Eigen/Dense>

namespace utils {
Eigen::MatrixXd finite_diff(const Eigen::Ref<Eigen::ArrayXd> &s, int M, int n_s);
};

#endif
