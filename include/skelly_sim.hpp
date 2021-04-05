#ifndef SKELLYSIM_HPP
#define SKELLYSIM_HPP

#include <toml.hpp>

#define EIGEN_USE_MKL_ALL
#define MKL_DIRECT_CALL

#include <Eigen/Core>
typedef Eigen::Map<Eigen::VectorXd> VectorMap;
typedef Eigen::Map<const Eigen::VectorXd> CVectorMap;
typedef Eigen::Map<Eigen::ArrayXd> ArrayMap;
typedef Eigen::Map<const Eigen::ArrayXd> CArrayMap;
typedef Eigen::Map<Eigen::MatrixXd> MatrixMap;
typedef Eigen::Map<const Eigen::MatrixXd> CMatrixMap;
typedef const Eigen::Ref<const Eigen::ArrayXd> ArrayRef;
typedef const Eigen::Ref<const Eigen::VectorXd> VectorRef;
typedef const Eigen::Ref<const Eigen::MatrixXd> MatrixRef;

#endif
